"""Popcon mode: a batched 2-D operating-space scan (POPCON).

The mode sweeps a grid over two scalar axis variables (e.g. average electron
density and temperature) and computes, for every grid point simultaneously, a
consistent operating point with the supplied inputs held exactly as given and
the axis values pinned to the grid coordinates.

The whole grid is one batched computation: every scalar carries a leading
``(N, 1)`` grid dimension and every profile ``(N, P)``, so each provider
relation of the compiled completion plan evaluates **once** on arrays instead
of once per point, and the free coupled cores (e.g. ``P_aux``) are solved by
one damped Gauss-Newton iteration vectorised across the grid.  Relations
whose implementation refuses arrays are evaluated point-wise inside the same
replay; a relation that produces non-finite values at some points poisons
only those points (NaN propagation), never the grid.

Validity never comes from the batched computation itself: every point is
individually certified by the same per-point machinery used everywhere else
(:func:`_certify` over the **certification cone** -- the requested outputs
plus everything connecting them to the inputs).  A point whose scenario
cannot satisfy the cone within tolerance fails certification and is masked,
never bent; no input is ever moved.  Reconciling a certified point is a
no-op by reconcile's already-verified short-circuit.

The system is compiled once (the active relation set depends only on *which*
variables are supplied, never on their numeric values); the core starts come
from the compile-time seeding oracle at the grid midpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._common import new_result, reject_unknown_options


def parse_axis(spec: Any, registry: Any) -> tuple[str, np.ndarray]:
    """Parse one popcon axis spec into ``(canonical_name, grid_values)``.

    Accepts a mapping with ``variable`` (or ``name``) plus either explicit
    ``values`` or ``start``/``stop``/``num`` (optional ``spacing`` of
    ``"linear"`` or ``"log"``), or the tuple shorthand
    ``(name, start, stop, num)``.  Values are in the variable's canonical
    unit.  The variable must be a scalar.
    """
    if isinstance(spec, Mapping):
        raw_name = spec.get("variable", spec.get("name"))
        if raw_name is None:
            raise ValueError("Axis spec needs a 'variable' key.")
        unknown = set(spec) - {"variable", "name", "values", "start", "stop", "num", "spacing"}
        if unknown:
            raise ValueError(f"Unknown axis spec key(s): {', '.join(sorted(unknown))}.")
        if spec.get("values") is not None:
            values = np.asarray(spec["values"], dtype=float).reshape(-1)
        else:
            values = _spaced_values(spec.get("start"), spec.get("stop"), spec.get("num"), spec.get("spacing", "linear"), raw_name)
    elif isinstance(spec, Sequence) and not isinstance(spec, str) and len(spec) == 4:
        raw_name, start, stop, num = spec
        values = _spaced_values(start, stop, num, "linear", raw_name)
    else:
        raise ValueError(
            f"Axis spec {spec!r} is not a mapping or a (name, start, stop, num) tuple."
        )
    name = registry.resolve(str(raw_name))
    if registry.get(name).shape != 0:
        raise ValueError(f"Axis variable {name!r} is a profile; popcon axes must be scalars.")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Axis {name!r} has an empty or non-finite grid.")
    return name, values


def _spaced_values(start: Any, stop: Any, num: Any, spacing: str, raw_name: Any) -> np.ndarray:
    if start is None or stop is None or num is None:
        raise ValueError(f"Axis {raw_name!r} needs 'values' or 'start'/'stop'/'num'.")
    if spacing == "linear":
        return np.linspace(float(start), float(stop), int(num))
    if spacing == "log":
        return np.geomspace(float(start), float(stop), int(num))
    raise ValueError(f"Axis {raw_name!r} spacing must be 'linear' or 'log', got {spacing!r}.")


def _relation_variable_names(system: Any, rel: Any) -> set[str]:
    """All registry variables one relation reads or writes, constants included.

    Constants that are themselves produced variables (e.g. per-channel
    ``P_fus_*`` powers read as constants) are real dependencies of the cone.
    """
    names = set(rel.variables)
    names.update(c for c in getattr(rel, "constant_names", ()) if c in system.variable_registry)
    return names


def certification_cone(system: Any, targets: Sequence[str]) -> tuple[list[Any], set[str]]:
    """Return the cone of relations/variables connecting ``targets`` to the inputs.

    Fixpoint closure over the compiled active system: providers and
    output-producing relations of needed variables pull in their full
    variable set, coupled structural blocks join wholesale, and enforced
    relations entirely inside the cone are added as consistency connectors.
    Certifying exactly this set is the popcon consistency guarantee: every
    relation among the computed quantities holds; variables outside the cone
    stay unconstrained scenario freedom.
    """
    provider_of: dict[str, Any] = {}
    provider_of.update(system.default_provider_by_output)
    provider_of.update(system.derived_provider_by_output)
    blocks = [frozenset(block) for block in system.structural_blocks]
    # Free-core components join wholesale: a free core in the cone (e.g.
    # P_aux) is only *determined* by its whole component's relations, so
    # certifying the core without them would certify an arbitrary number.
    components = _free_cores(system)
    needed: set[str] = set(targets)
    cone_rels: dict[str, Any] = {}

    changed = True
    while changed:
        changed = False
        for name in tuple(needed):
            rel = provider_of.get(name)
            if rel is not None and rel.name not in cone_rels:
                cone_rels[rel.name] = rel
                needed |= _relation_variable_names(system, rel)
                changed = True
        for rel in system.relations:
            if rel.name in cone_rels:
                continue
            variables = _relation_variable_names(system, rel)
            produces_needed = bool(set(rel.output_names or ()) & needed)
            inside_cone = rel.enforce and variables <= needed
            if produces_needed or inside_cone:
                cone_rels[rel.name] = rel
                needed |= variables
                changed = True
        for block in blocks:
            if block & needed and not block <= needed:
                needed |= block
                changed = True
        for unknowns, rels in components:
            if not (set(unknowns) & needed):
                continue
            for rel in rels:
                if rel.name not in cone_rels:
                    cone_rels[rel.name] = rel
                    needed |= _relation_variable_names(system, rel)
                    changed = True
            if not set(unknowns) <= needed:
                needed |= set(unknowns)
                changed = True
    return list(cone_rels.values()), needed


def _free_cores(system: Any) -> list[tuple[tuple[str, ...], list[Any]]]:
    """Independent components of free coupled unknowns with their relations.

    Free cores are the block-core variables and structural-block members that
    are not supplied/fixed -- the quantities reconcile would pack (e.g.
    P_aux, tau_E, P_loss).  They are grouped into connected components (two
    unknowns connect when an enforced relation touches both) so each
    component gets its own small Gauss-Newton; block members with a provider
    are recomputed by the completion replay and only the provider-less true
    cores are searched.
    """
    unknowns: set[str] = {
        name for name, role in system.variable_roles.items() if role == "core"
    }
    for block in system.structural_blocks:
        unknowns.update(block)
    unknowns -= set(system.fixed)
    unknowns = {name for name in unknowns if system.inputs.get(name) is None}

    touching = [
        rel
        for rel in system.relations
        if rel.enforce and any(name in unknowns for name in rel.variables)
    ]
    parent = {name: name for name in unknowns}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for rel in touching:
        members = [name for name in rel.variables if name in unknowns]
        for other in members[1:]:
            parent[find(other)] = find(members[0])

    grouped: dict[str, set[str]] = {}
    for name in unknowns:
        grouped.setdefault(find(name), set()).add(name)
    components: list[tuple[tuple[str, ...], list[Any]]] = []
    for members in grouped.values():
        rels = [rel for rel in touching if any(name in members for name in rel.variables)]
        components.append((tuple(sorted(members)), rels))
    components.sort(key=lambda item: item[0])
    return components


def _certify(system: Any, ns: dict[str, Any], cone_rels: Sequence[Any], cone_vars: set[str]) -> tuple[bool, dict[str, Any], list[str]]:
    """Certify the cone relations on one per-point solver namespace.

    Returns ``(verified, relation_status, errors)``.  Errors cover fixed
    variables drifting off their pinned values and domain violations of cone
    variables; a relation that cannot even be evaluated fails certification.
    """
    errors = list(system.fixed_value_errors(ns))
    errors.extend(system.domain_errors({name: ns.get(name) for name in cone_vars}))
    relation_status: dict[str, Any] = {}
    for rel in cone_rels:
        if not rel.enforce:
            continue
        try:
            status, _residual = system.relation_status_and_residual(rel, system.relation_evaluation_values(rel, ns))
        except Exception as exc:
            status = {"relation": rel.name, "verified": False, "enforced": True, "errors": [str(exc)], "warnings": []}
        relation_status[rel.name] = status
    verified = not errors and all(status.get("verified", False) for status in relation_status.values())
    return verified, relation_status, errors


# ── Batched namespace machinery ──────────────────────────────────────────
#
# Shape discipline: scalars are (N, 1), profiles are (N, P), the rho grid
# stays (P,).  Scalar x rho expressions inside relation code then broadcast
# to (N, P) exactly as scalar x rho broadcasts to (P,) in the per-point
# world, and profile reductions (trapezoid over the last axis) produce (N,)
# which the write-time coercion restores to (N, 1).


def _coerce_batched(value: Any, shape: int, n: int, profile_size: int) -> np.ndarray | None:
    """Coerce one relation output to the batched layout, or ``None`` if it
    cannot be interpreted for the registry shape."""
    arr = np.asarray(value, dtype=float)
    if shape == 0:
        if arr.ndim == 0:
            return np.full((n, 1), float(arr))
        if arr.shape in ((n,), (n, 1)):
            return arr.reshape(n, 1)
        if arr.shape in ((1,), (1, 1)):
            return np.full((n, 1), float(arr.reshape(-1)[0]))
        return None
    if arr.ndim == 0:
        return np.full((n, profile_size), float(arr))
    if arr.shape == (profile_size,):
        return np.broadcast_to(arr, (n, profile_size)).copy()
    if arr.shape == (n, profile_size):
        return arr
    if arr.shape in ((n, 1), (n,)):
        return np.broadcast_to(arr.reshape(n, 1), (n, profile_size)).copy()
    return None


def _slice_point(system: Any, ns: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    """Extract one grid point's per-point solver namespace from the batch."""
    out: dict[str, Any] = {}
    for name, arr in ns.items():
        if arr is None:
            continue
        if arr.ndim == 2 and arr.shape[1] == 1:
            out[name] = float(arr[index, 0])
        elif arr.ndim == 2:
            out[name] = np.ascontiguousarray(arr[index])
        elif arr.ndim == 1 and arr.shape[0] != 1:
            # Unbatched shared arrays (the rho grid) pass through whole.
            out[name] = arr
        else:
            out[name] = float(np.asarray(arr).reshape(-1)[0])
    return out


def _eval_pointwise(system: Any, rel: Any, ns: Mapping[str, np.ndarray], n: int, out_specs: Sequence[tuple[str, Any]]) -> dict[str, np.ndarray]:
    """Evaluate one relation point-by-point; failed points contribute NaN."""
    profile_size = system.profile_size
    collected: dict[str, list[Any]] = {name: [] for name, _spec in out_specs}
    with np.errstate(all="ignore"):
        for index in range(n):
            try:
                mapped = rel.output_map(rel.evaluate(_slice_point(system, ns, index)))
            except Exception:
                mapped = {}
            for name, _spec in out_specs:
                collected[name].append(mapped.get(name))
    out: dict[str, np.ndarray] = {}
    for name, spec in out_specs:
        if spec.shape == 0:
            out[name] = np.asarray(
                [np.nan if v is None else float(np.asarray(v, dtype=float).reshape(-1)[0]) for v in collected[name]],
                dtype=float,
            ).reshape(n, 1)
        else:
            rows = []
            for v in collected[name]:
                if v is None:
                    rows.append(np.full(profile_size, np.nan))
                else:
                    arr = np.asarray(v, dtype=float)
                    rows.append(np.broadcast_to(arr, (profile_size,)).astype(float) if arr.ndim == 0 else arr.reshape(-1)[:profile_size])
            out[name] = np.vstack(rows)
    return out


def _matches_point(batched: np.ndarray, reference: Any, index: int) -> bool:
    """Compare one batched row against a per-point recomputation."""
    if reference is None:
        return False
    ref = np.asarray(reference, dtype=float).reshape(-1)
    row = np.asarray(batched[index], dtype=float).reshape(-1)
    if ref.size == 1 and row.size >= 1:
        ref = np.broadcast_to(ref, row.shape)
    if ref.shape != row.shape:
        return False
    return bool(np.allclose(row, ref, rtol=1.0e-9, atol=0.0, equal_nan=True))


def _relation_outputs(
    system: Any,
    rel: Any,
    ns: dict[str, np.ndarray],
    n: int,
    out_specs: Sequence[tuple[str, Any]],
    trust: dict[str, str],
) -> dict[str, np.ndarray]:
    """Batched outputs of one relation, coerced to the batched layout.

    Tries the vectorised call first (identical to the per-point completion
    call, just with array values).  The result is accepted only if every
    requested output coerces to the expected batched shape AND a two-point
    spot check (first and last grid point recomputed per-point) matches --
    this catches implementations whose internal reductions broadcast into
    outer products or that silently collapse the batch to one point's value.
    Anything else falls back to the point-by-point loop, where a failing
    point contributes NaN and poisons only itself.  The verdict is cached per
    relation for the scan (``trust``): broadcastability is a structural
    property of the implementation, so the spot check runs once, not once
    per solver iteration.
    """
    profile_size = system.profile_size
    verdict = trust.get(rel.name)
    if verdict == "pointwise":
        return _eval_pointwise(system, rel, ns, n, out_specs)
    with np.errstate(all="ignore"):
        try:
            mapped = rel.output_map(rel.evaluate(ns))
        except Exception:
            mapped = None
    if mapped is not None:
        coerced: dict[str, np.ndarray] = {}
        for name, spec in out_specs:
            arr = None if mapped.get(name) is None else _coerce_batched(mapped[name], spec.shape, n, profile_size)
            if arr is None:
                coerced = {}
                break
            coerced[name] = arr
        if coerced:
            if verdict == "batched":
                return coerced
            checks = (0, n - 1) if n > 1 else (0,)
            reference: dict[int, dict[str, Any]] = {}
            with np.errstate(all="ignore"):
                for index in checks:
                    try:
                        reference[index] = rel.output_map(rel.evaluate(_slice_point(system, ns, index)))
                    except Exception:
                        reference[index] = {}
            if all(
                _matches_point(coerced[name], reference[index].get(name), index)
                for name, _spec in out_specs
                for index in checks
            ):
                trust[rel.name] = "batched"
                return coerced
    trust[rel.name] = "pointwise"
    return _eval_pointwise(system, rel, ns, n, out_specs)


def _replay_completion(system: Any, ns: dict[str, np.ndarray], n: int, trust: dict[str, str]) -> None:
    """Run the compiled completion plan on the batched namespace, in place.

    Mirrors :meth:`RelationSystem._apply_completion_providers`: providers in
    dependency order, explicit providers recompute, defaults fill only
    missing outputs, cyclic plans iterate to a fixed point.  The only
    difference is the write-time coercion to the batched layout instead of
    the per-point spec coercion; solver-domain projection is skipped here
    because certification re-checks domains per point.
    """
    held = {name for name in system.inputs if system.inputs.get(name) is not None} | set(system.fixed)
    for _pass in range(system._completion_passes):
        changed = False
        # Re-level shape-controlled profiles from their (possibly provider-
        # updated) scalar averages, mirroring the profile stage of
        # RelationSystem.complete().  A supplied profile is level-free: its
        # shape is fixed but its level tracks its average, which may itself be
        # a scan axis or be derived from one (e.g. T_i_avg = T_e_avg), so it
        # must be rebuilt each pass rather than frozen at the compile midpoint.
        for name, avg_name, shape, fixed_value in system._profile_specs:
            if fixed_value is None and avg_name is not None and ns.get(avg_name) is not None:
                relevelled = ns[avg_name] * np.asarray(shape, dtype=float)
                if ns.get(name) is None or not np.array_equal(ns[name], relevelled):
                    ns[name] = relevelled
                    changed = True
        for rel, only_missing, input_names, outs in system._provider_plan:
            if any(ns.get(name) is None for name in input_names):
                continue
            # Supplied inputs are the scenario and are held exactly: a
            # provider whose output is a held input never overwrites it and
            # becomes a pure consistency check at certification.
            writable = [
                (out_name, spec)
                for out_name, spec in outs
                if out_name not in held and not (only_missing and ns.get(out_name) is not None)
            ]
            if not writable:
                continue
            outputs = _relation_outputs(system, rel, ns, n, writable, trust)
            for out_name, _spec in writable:
                arr = outputs.get(out_name)
                if arr is None:
                    continue
                old = ns.get(out_name)
                ns[out_name] = arr
                if old is None or old.shape != arr.shape or not np.allclose(old, arr, rtol=0.0, atol=1.0e-300, equal_nan=True):
                    changed = True
        if not changed:
            break


def _batched_base(system: Any, n: int) -> dict[str, np.ndarray]:
    """Batched pure-input namespace: inputs broadcast, profiles and constants
    applied exactly as the per-point completion stages do."""
    profile_size = system.profile_size
    ns: dict[str, np.ndarray] = {}
    for name, value in system.input_values().items():
        spec = system.variable_registry.get(name)
        if name == "rho":
            ns[name] = np.asarray(value, dtype=float)  # shared, unbatched
            continue
        arr = _coerce_batched(value, spec.shape, n, profile_size)
        if arr is not None:
            ns[name] = arr
    # Profile stage: fixed profiles verbatim, shape-controlled from averages.
    for name, avg_name, shape, fixed_value in system._profile_specs:
        if fixed_value is not None:
            ns[name] = np.broadcast_to(np.asarray(fixed_value, dtype=float), (n, profile_size)).copy()
            continue
        if avg_name is None or ns.get(avg_name) is None:
            continue
        ns[name] = ns[avg_name] * np.asarray(shape, dtype=float)
    # Constant-default stage.
    for name, value in system._constant_defaults_solver.items():
        if ns.get(name) is None:
            spec = system.variable_registry.get(name)
            arr = _coerce_batched(value, spec.shape, n, profile_size)
            if arr is not None:
                ns[name] = arr
    return ns


def _outputless_rows(system: Any, rel: Any, ns: dict[str, np.ndarray], n: int, trust: dict[str, str]) -> np.ndarray:
    """Batched raw-residual rows of an outputless (adirectional) relation.

    ``evaluate()`` of such a relation *is* its scaled residual (e.g. the
    energy-confinement balance).  Same batched-first strategy with a
    two-point spot check and per-point fallback as :func:`_relation_outputs`.
    """
    key = rel.name
    if trust.get(key) != "pointwise":
        with np.errstate(all="ignore"):
            try:
                arr = np.asarray(rel.evaluate(ns), dtype=float)
            except Exception:
                arr = None
        rows = None
        if arr is not None:
            if arr.ndim == 0:
                rows = np.full((n, 1), float(arr))
            elif arr.shape in ((n,), (n, 1)):
                rows = arr.reshape(n, 1)
        if rows is not None:
            if trust.get(key) == "batched":
                return rows
            checks = (0, n - 1) if n > 1 else (0,)
            with np.errstate(all="ignore"):
                try:
                    matches = all(
                        np.allclose(rows[index], np.asarray(rel.evaluate(_slice_point(system, ns, index)), dtype=float).reshape(-1), rtol=1.0e-9, atol=0.0, equal_nan=True)
                        for index in checks
                    )
                except Exception:
                    matches = False
            if matches:
                trust[key] = "batched"
                return rows
        trust[key] = "pointwise"
    values = np.full((n, 1), np.nan)
    with np.errstate(all="ignore"):
        for index in range(n):
            try:
                values[index, 0] = float(np.asarray(rel.evaluate(_slice_point(system, ns, index)), dtype=float).reshape(-1)[0])
            except Exception:
                pass
    return values


def _component_residual(system: Any, ns: dict[str, np.ndarray], rels: Sequence[Any], n: int, trust: dict[str, str]) -> np.ndarray:
    """Batched residual rows for one core component, ``(N, rows)``.

    Relations with declared outputs compare them against the namespace with
    a per-point relative scale (tautologically zero for the component's own
    providers, informative for cross-checks); outputless relations
    contribute their raw adirectional residual.  This is a solving aid only
    -- certification applies the canonical tolerance semantics afterwards.
    """
    columns: list[np.ndarray] = []
    for rel in rels:
        if not rel.output_names:
            if all(ns.get(name) is not None for name in rel.variables):
                columns.append(_outputless_rows(system, rel, ns, n, trust))
            continue
        out_specs = [
            (out_name, system.variable_registry.get(out_name))
            for out_name in rel.output_names
            if ns.get(out_name) is not None
        ]
        if not out_specs:
            continue
        outputs = _relation_outputs(system, rel, ns, n, out_specs, trust)
        with np.errstate(all="ignore"):
            for out_name, _spec in out_specs:
                current = ns[out_name]
                arr = outputs.get(out_name)
                if arr is None:
                    columns.append(np.full((n, 1), 1.0e3))
                    continue
                scale = np.maximum(np.abs(current), np.maximum(np.abs(arr), 1.0e-30))
                rows = (current - arr) / scale
                columns.append(rows.reshape(n, -1))
    if not columns:
        return np.zeros((n, 0))
    out = np.concatenate(columns, axis=1)
    return np.nan_to_num(out, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)


def _solve_cores_batched(
    system: Any,
    ns: dict[str, np.ndarray],
    components: Sequence[tuple[tuple[str, ...], Sequence[Any]]],
    n: int,
    trust: dict[str, str],
    *,
    max_iterations: int = 25,
    target: float = 1.0e-9,
) -> None:
    """Damped Gauss-Newton over the provider-less true cores, vectorised
    across the grid, with the completion replay refreshing derived members
    between iterations.  Points that fail to converge keep their best state;
    certification is the arbiter."""
    provided = set(system.default_provider_by_output) | set(system.derived_provider_by_output)
    for unknowns, rels in components:
        cores = [name for name in unknowns if name not in provided and system.variable_registry.get(name).shape == 0]
        if not cores:
            continue
        bounds = [system.spec_of(name).solver_bounds for name in cores]
        # Start from the compile-time seeding-oracle guesses (grid midpoint).
        for name in cores:
            if ns.get(name) is None:
                try:
                    start = float(system.initial_value(name))
                except Exception:
                    start = 1.0
                ns[name] = np.full((n, 1), start)

        def clamp(index: int, arr: np.ndarray) -> np.ndarray:
            low, high = bounds[index]
            return np.clip(arr, low if np.isfinite(low) else -np.inf, high if np.isfinite(high) else np.inf)

        _replay_completion(system, ns, n, trust)
        residual = _component_residual(system, ns, rels, n, trust)
        if residual.shape[1] == 0:
            # No output-bearing enforced relation constrains this component;
            # its cores keep their starts and certification arbitrates.
            continue
        floors = []
        for name in cores:
            try:
                floors.append(float(system.spec_of(name).tolerance_floor(*system.tols_of(name))))
            except Exception:
                floors.append(1.0)
        best = np.max(np.abs(residual), axis=1)
        damping = np.full(n, 1.0)
        for _iteration in range(max_iterations):
            active = best > target
            if not active.any():
                break
            k = len(cores)
            base_values = {name: ns[name].copy() for name in cores}
            # Per-point core magnitude references: the Gauss-Newton runs in
            # normalised units so JtJ is O(1) and the Levenberg damping is
            # meaningful regardless of the cores' physical scale (watts vs
            # seconds).
            refs = [np.maximum(np.abs(base_values[name]), floors[j]) for j, name in enumerate(cores)]
            jacobian = np.zeros((n, residual.shape[1], k))
            dz = 1.0e-6
            for j, name in enumerate(cores):
                ns[name] = clamp(j, base_values[name] + dz * refs[j])
                actual = (ns[name] - base_values[name]) / refs[j]
                actual[actual == 0.0] = 1.0e-300
                _replay_completion(system, ns, n, trust)
                jacobian[:, :, j] = (_component_residual(system, ns, rels, n, trust) - residual) / actual
                ns[name] = base_values[name]
            jt = np.swapaxes(jacobian, 1, 2)
            jtj = jt @ jacobian
            jtr = (jt @ residual[:, :, None])[:, :, 0]
            identity = np.eye(k)
            lam = (1.0e-10 + (1.0 - damping) * 1.0e-2)[:, None, None]
            try:
                delta = np.linalg.solve(jtj + lam * identity, jtr[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                delta = np.zeros((n, k))
            delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            # Damped update on active points only; backtrack points that worsen.
            improved = np.zeros(n, dtype=bool)
            scale = damping.copy()
            for _backtrack in range(4):
                trial_mask = active & ~improved
                if not trial_mask.any():
                    break
                for j, name in enumerate(cores):
                    trial = base_values[name].copy()
                    trial[trial_mask, 0] = base_values[name][trial_mask, 0] - scale[trial_mask] * delta[trial_mask, j] * refs[j][trial_mask, 0]
                    ns[name] = clamp(j, trial)
                _replay_completion(system, ns, n, trust)
                trial_residual = _component_residual(system, ns, rels, n, trust)
                trial_best = np.max(np.abs(trial_residual), axis=1)
                newly = trial_mask & (trial_best < best)
                improved |= newly
                best = np.where(newly, trial_best, best)
                residual = np.where(newly[:, None], trial_residual, residual)
                for name in cores:
                    base_values[name][newly, 0] = ns[name][newly, 0]
                scale = np.where(improved, scale, scale * 0.25)
            for name in cores:
                ns[name] = base_values[name]
            damping = np.where(improved, np.minimum(damping * 2.0, 1.0), np.maximum(damping * 0.25, 1.0e-3))
            if not improved.any():
                break
        _replay_completion(system, ns, n, trust)


def run(
    system: Any,
    *,
    x: Any = None,
    y: Any = None,
    outputs: Sequence[str] | None = None,
    verbose: int = 0,
    **_unused: Any,
) -> dict[str, Any]:
    """Batched 2-D scan: evaluate, solve the cores, certify every point.

    Args:
        x: X-axis spec (see :func:`parse_axis`).
        y: Y-axis spec.
        outputs: Scalar variables to collect; defaults to every active
            scalar.  They also define the certification cone, so a lean list
            certifies (and guarantees) exactly the quantities of interest
            plus everything connecting them.
        verbose: Nonzero prints stage progress.

    Returns:
        Mode result with a ``"popcon"`` payload: ``x``/``y`` axis arrays,
        ``fields`` (one ``(ny, nx)`` float array per output, ``NaN`` where a
        point failed certification), the boolean ``success`` grid and a
        ``failures`` list naming each failed point's violated relations.
    """
    self = system
    result = new_result(self, "popcon")
    if reject_unknown_options(result, _unused):
        return result
    if x is None or y is None:
        result["errors"].append("popcon requires 'x' and 'y' axis specs.")
        result["termination"] = "invalid options"
        return result
    try:
        x_name, x_values = parse_axis(x, self.variable_registry)
        y_name, y_values = parse_axis(y, self.variable_registry)
    except Exception as exc:
        result["errors"].append(str(exc))
        result["termination"] = "invalid options"
        return result
    if x_name == y_name:
        result["errors"].append(f"popcon axes must differ; both are {x_name!r}.")
        result["termination"] = "invalid options"
        return result

    # Full pre-scan state, restored verbatim when the scan finishes.
    saved_inputs = dict(self.inputs)
    saved_values = dict(self.values)
    saved_fixed = set(self.fixed)

    nx, ny = int(x_values.size), int(y_values.size)
    n = nx * ny
    grid_x, grid_y = np.meshgrid(x_values, y_values)  # (ny, nx), rows = y
    flat_x, flat_y = grid_x.reshape(-1), grid_y.reshape(-1)

    try:
        # Pin the axes to the user-requested grid coordinates (the scan
        # values are user-specified; nothing else is fixed beyond the
        # reactor's own declarations) and compile once at the grid midpoint
        # so the axes count as supplied/fixed and the seeding oracle leaves
        # mid-grid core guesses.
        for name, values in ((x_name, x_values), (y_name, y_values)):
            midpoint = float(values[values.size // 2])
            self.inputs[name] = midpoint
            self.values[name] = midpoint
            self.fixed.add(name)
        self.compile()

        if outputs is None:
            output_names = sorted(
                name
                for name in self.active_variable_names
                if self.variable_registry.get(name).shape == 0 and name not in (x_name, y_name)
            )
        else:
            output_names = [self.variable_registry.resolve(str(name)) for name in outputs]
            # A requested output the compiled system can neither read nor
            # derive stays NaN at every point without failing certification
            # (its cone is empty); surface that instead of a silent blank map.
            underivable = [
                name
                for name in output_names
                if name not in self.active_variable_names and self.inputs.get(name) is None
            ]
            if underivable:
                result["warnings"].append(
                    "Requested output(s) not derivable by the compiled system (left NaN): "
                    + ", ".join(underivable)
                    + ". They are inactive -- likely missing upstream inputs."
                )

        cone_rels, cone_vars = certification_cone(self, (*output_names, x_name, y_name))
        components = _free_cores(self)

        # Batched namespace: inputs broadcast, axes set to the flattened grid.
        trust: dict[str, str] = {}
        ns = _batched_base(self, n)
        ns[x_name] = flat_x.reshape(n, 1)
        ns[y_name] = flat_y.reshape(n, 1)
        if verbose:
            print(f"popcon batch: {n} points, {len(self._provider_plan)} providers, {len(components)} core components")
        _replay_completion(self, ns, n, trust)
        _solve_cores_batched(self, ns, components, n, trust)

        # Per-point certification through the same machinery as every other
        # mode -- the batched computation only produced candidates.
        fields = {name: np.full((ny, nx), np.nan) for name in output_names}
        success = np.zeros((ny, nx), dtype=bool)
        failures: list[dict[str, Any]] = []
        for index in range(n):
            iy, ix = divmod(index, nx)
            point = _slice_point(self, ns, index)
            # The fixed-drift check compares against the pinned inputs, so
            # the axis inputs must hold this point's grid coordinates.
            self.inputs[x_name] = float(flat_x[index])
            self.inputs[y_name] = float(flat_y[index])
            verified, relation_status, errors = _certify(self, point, cone_rels, cone_vars)
            success[iy, ix] = verified
            if verified:
                for name in output_names:
                    value = point.get(name)
                    if value is not None:
                        try:
                            public = self.public_value(name, value)
                            fields[name][iy, ix] = float(np.asarray(public, dtype=float).reshape(-1)[0])
                        except (TypeError, ValueError):
                            pass
            else:
                failed = [name for name, status in relation_status.items() if not status.get("verified", False)]
                detail = ", ".join(failed[:4]) if failed else "; ".join(errors[:2])
                failures.append({"ix": int(ix), "iy": int(iy), "termination": f"certification failed: {detail}"})
        if verbose:
            print(f"popcon batch: {n - len(failures)}/{n} points certified")
    finally:
        # Leave the system exactly as found, whatever happened mid-scan.
        self.inputs.clear()
        self.inputs.update(saved_inputs)
        self.values.clear()
        self.values.update(saved_values)
        self.fixed.clear()
        self.fixed.update(saved_fixed)
        self.compile()

    n_failed = len(failures)
    result.update(
        {
            # Infeasible regions are normal in a popcon; the scan succeeds
            # when it ran and at least one operating point certified.
            "success": n_failed < n,
            "termination": f"popcon scan completed: {n - n_failed}/{n} points solved",
            "n_points": n,
            "n_failed": n_failed,
            "popcon": {
                "x": {"name": x_name, "values": x_values},
                "y": {"name": y_name, "values": y_values},
                "fields": fields,
                "success": success,
                "failures": failures,
            },
        }
    )
    return result
