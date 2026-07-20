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

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fusdb.relationsystem import (
    apply_completion_providers_batched,
    coerce_batched,
    relation_outputs,
    slice_point,
)

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
        name for name, role in system.variable_roles.items() if role in ("core", "packed")
    }
    for block in system.structural_blocks:
        unknowns.update(block)
    unknowns -= set(system.fixed)
    # Packed supplied inputs (movement-anchored scenario values) are not scan
    # unknowns -- only the seeded free cores (e.g. a peaking factor closed by
    # a coupled cycle) join the batched core search.
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


# The batched namespace machinery (coerce/slice/trust/replay) lives next to
# the per-point completion loop it mirrors -- see the "Batched completion"
# section of fusdb.relationsystem for the shape discipline and the deliberate
# differences from RelationSystem._apply_completion_providers.

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
        arr = coerce_batched(value, spec.shape, n, profile_size)
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
            arr = coerce_batched(value, spec.shape, n, profile_size)
            if arr is not None:
                ns[name] = arr
    return ns


def _outputless_rows(system: Any, rel: Any, ns: dict[str, np.ndarray], n: int, trust: dict[str, str]) -> np.ndarray:
    """Batched raw-residual rows of an outputless (adirectional) relation.

    ``evaluate()`` of such a relation *is* its scaled residual (e.g. the
    energy-confinement balance).  Same batched-first strategy with a
    two-point spot check and per-point fallback as :func:`relation_outputs`.
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
            # Spot-check on finite rows: a NaN row is a poisoned grid point
            # whose per-point recomputation may raise instead of matching.
            finite = np.flatnonzero(np.isfinite(rows[:, 0]))
            if finite.size:
                checks = tuple(dict.fromkeys((int(finite[0]), int(finite[-1]))))
            else:
                checks = (0, n - 1) if n > 1 else (0,)
            with np.errstate(all="ignore"):
                try:
                    matches = all(
                        np.allclose(rows[index], np.asarray(rel.evaluate(slice_point(system, ns, index)), dtype=float).reshape(-1), rtol=1.0e-9, atol=0.0, equal_nan=True)
                        for index in checks
                    )
                except Exception:
                    matches = False
            if matches:
                trust[key] = "batched"
                return rows
        trust[key] = "pointwise"
    needed = set(rel.variables) | set(getattr(rel, "constant_names", ()))
    values = np.full((n, 1), np.nan)
    with np.errstate(all="ignore"):
        for index in range(n):
            try:
                values[index, 0] = float(np.asarray(rel.evaluate(slice_point(system, ns, index, needed)), dtype=float).reshape(-1)[0])
            except Exception:
                pass
    return values


def _fill_packed_producibles(
    system: Any, ns: dict[str, np.ndarray], n: int, trust: dict[str, str], dirty: set[str] | None = None
) -> set[str]:
    """Forward-fill packed unknowns that completion does not own each pass.

    Two kinds of packed variable are refreshed here (a packed var is a solver
    unknown reconcile would move, so the compile gives it no *mandatory*
    provider):

    * **profile-valued** ones (e.g. ``n_i`` under a soft composition) -- the
      batched scan cannot pack profiles, so any active relation that outputs
      them is evaluated forward;
    * **scalars behind a default provider** (e.g. ``density_peaking`` behind
      the Angioni scaling) -- their default provider is ``only_missing``, so
      completion freezes them at the seed; recomputing them forward lets the
      value track its cycle sibling that *is* a Gauss-Newton core (``beta_T``),
      so the batched peaking cycle converges to the same fixed point reconcile
      does (otherwise it stalls one Gauss-Seidel step short, ~2% off).

    Scalar packed unknowns with no provider are the Gauss-Newton cores and are
    left untouched -- the solver owns them.

    ``dirty`` follows the :func:`apply_completion_providers_batched` contract:
    only relations reading a touched variable (or missing an output) are
    evaluated; ``None`` evaluates everything.  Returns the names written.
    """
    roles = system.variable_roles
    default_produced = set(system.default_provider_by_output)
    held = {name for name in system.inputs if system.inputs.get(name) is not None} | set(system.fixed)
    wrote: set[str] = set()
    for rel in system.relations:
        outs = [
            (out_name, system.variable_registry.get(out_name))
            for out_name in rel.output_names
            if roles.get(out_name) == "packed"
            and out_name not in held
            and (system.variable_registry.get(out_name).shape != 0 or out_name in default_produced)
        ]
        if not outs or any(ns.get(name) is None for name in rel.input_names):
            continue
        if (
            dirty is not None
            and all(ns.get(out_name) is not None for out_name, _spec in outs)
            and dirty.isdisjoint(rel.input_names)
            and dirty.isdisjoint(getattr(rel, "constant_names", ()))
        ):
            continue
        outputs = relation_outputs(system, rel, ns, n, outs, trust)
        for out_name, _spec in outs:
            arr = outputs.get(out_name)
            if arr is None:
                continue
            old = ns.get(out_name)
            if old is None or old.shape != arr.shape or not np.allclose(old, arr, rtol=1e-12, atol=0.0, equal_nan=True):
                ns[out_name] = arr
                wrote.add(out_name)
    return wrote


def _refresh_batched(
    system: Any, ns: dict[str, np.ndarray], n: int, trust: dict[str, str], dirty: set[str] | None = None
) -> None:
    """Completion replay plus the packed-profile fill, iterated to a fixpoint.

    The packed-profile/peaking fill and the completion replay feed each other
    (density_peaking -> n_e profile -> p_th -> beta_T -> density_peaking), so
    they are alternated until neither writes a change.  The cap is generous
    because the batched value must match a fully-coupled reconcile to solver
    precision (the popcon consistency guarantee), which a few Gauss-Seidel
    sweeps of a slowly-contracting cycle would miss.

    ``dirty`` names what the caller changed since the previous refresh
    (``None`` = everything); each stage passes on exactly what it wrote, so
    untouched provider chains are skipped end to end.
    """
    changed = apply_completion_providers_batched(system, ns, n, trust, dirty=dirty)
    fill_dirty = None if dirty is None else set(dirty) | changed
    for _pass in range(30):
        wrote = _fill_packed_producibles(system, ns, n, trust, dirty=fill_dirty)
        if not wrote:
            break
        changed = apply_completion_providers_batched(system, ns, n, trust, dirty=set(wrote))
        fill_dirty = set(wrote) | changed


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
        outputs = relation_outputs(system, rel, ns, n, out_specs, trust)
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
    # Every namespace write below is recorded in ``stale`` so each refresh
    # replays only the cones downstream of what actually moved (the caller's
    # full refresh established a consistent namespace before this solve).
    stale: set[str] = set()

    def refresh() -> None:
        _refresh_batched(system, ns, n, trust, dirty=set(stale))
        stale.clear()
    # Components are coupled through provided intermediates (a singleton core
    # like an average feeds the big power component's stored energy), so solve
    # small components first and sweep the whole set twice: the second sweep
    # re-solves every component against the others' settled values.
    ordered = sorted(components, key=lambda item: len(item[0]))
    for unknowns, rels in ordered * 2:
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
                stale.add(name)

        def clamp(index: int, arr: np.ndarray) -> np.ndarray:
            low, high = bounds[index]
            return np.clip(arr, low if np.isfinite(low) else -np.inf, high if np.isfinite(high) else np.inf)

        refresh()
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
                stale.add(name)
                actual = (ns[name] - base_values[name]) / refs[j]
                actual[actual == 0.0] = 1.0e-300
                refresh()
                jacobian[:, :, j] = (_component_residual(system, ns, rels, n, trust) - residual) / actual
                ns[name] = base_values[name]
                stale.add(name)
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
                    stale.add(name)
                refresh()
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
                stale.add(name)
            damping = np.where(improved, np.minimum(damping * 2.0, 1.0), np.maximum(damping * 0.25, 1.0e-3))
            if not improved.any():
                break
        refresh()


def _reconcile_point(
    self: Any,
    pins: Mapping[str, float],
    seed: Mapping[str, Any] | None,
    saved_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile one grid point as its own system, warm-started from ``seed``.

    The point is a full reconcile -- the same path (and therefore the same
    verification) any single scenario gets -- with the axes pinned and every
    other input at the reactor's declared value.  ``seed`` (a previously solved
    neighbour's values) is injected as the seeding oracle's initial guesses, so
    a continuation step starts essentially at its answer.

    The system is **not** recompiled per point: the active relation set depends
    only on *which* variables are supplied, never on their numeric values, and
    the axes are pinned for every cell -- so one compile (done by the caller)
    serves the whole grid.
    """
    self.inputs.clear()
    self.inputs.update(saved_inputs)
    self.inputs.update(pins)
    self.values.clear()
    self.values.update(self.inputs)
    if seed:
        # Warm start: every free (non-pinned) value the neighbour solved for.
        self.values.update(
            {name: value for name, value in seed.items() if name not in pins and value is not None}
        )
        self.initial_guesses.update(
            {name: value for name, value in seed.items() if name not in self.fixed and value is not None}
        )
    return self.run("reconcile")


def _point_fields(system: Any, output_names: Sequence[str] | None) -> dict[str, float]:
    """Extract the requested (or all scalar) solved values as public floats."""
    values = system.values
    names = output_names if output_names is not None else [
        name for name in values if system.variable_registry.get(name).shape == 0
    ]
    out: dict[str, float] = {}
    for name in names:
        value = values.get(name)
        if value is None:
            continue
        try:
            out[name] = float(np.asarray(system.public_value(name, value), dtype=float).reshape(-1)[0])
        except (TypeError, ValueError):
            continue
    return out


def _system_spec(system: Any) -> dict[str, Any]:
    """Picklable recipe to rebuild an equivalent system in a worker process.

    Live systems cannot cross a process boundary (relation functions and parsed
    constraints do not pickle), so a worker receives the record-level state --
    supplied canonical values, fixed flags, tolerances -- plus registry relation
    *names* and the raw constraint spec, and rebuilds from the shared registry.
    """
    return {
        "name": system.name,
        "variables": [
            (
                name,
                system.inputs.get(name),
                name in system.fixed,
                system.rel_tols.get(name),
                system.abs_tols.get(name),
            )
            for name in sorted(system.rel_tols)
            if name != "rho"
        ],
        "relations": [rel.name for rel in system.candidate_primary_relations],
        "constraints": system.constraints_spec,
    }


def _rebuild_system(spec: Mapping[str, Any]) -> Any:
    from fusdb.registry import RELATIONS
    from fusdb.relationsystem import RelationSystem
    from fusdb.variable import Variable

    variables = [
        Variable(name, value=value, rel_tol=rel_tol, abs_tol=abs_tol, fixed=fixed)
        for name, value, fixed, rel_tol, abs_tol in spec["variables"]
    ]
    relations = [RELATIONS.get(name) for name in spec["relations"]]
    return RelationSystem(variables, relations, constraints=spec["constraints"], name=spec["name"])


def _march_column(
    system: Any,
    pinned_inputs: Mapping[str, Any],
    x_name: str,
    x_value: float,
    y_name: str,
    y_values: np.ndarray,
    iy0: int,
    seed: Mapping[str, Any] | None,
    output_names: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """March one column up and down from its reference-row cell.

    Each point warm-starts from the last *solved* point of its half-column
    (falling back to the column's row seed), so a failed cell cannot poison its
    children.  Returns one record per visited cell (the ``iy0`` cell itself
    belongs to the reference row and is not re-solved here).
    """
    records: list[dict[str, Any]] = []
    ny = int(y_values.size)
    for direction in (range(iy0 + 1, ny), range(iy0 - 1, -1, -1)):
        parent_seed = seed
        for iy in direction:
            pins = {x_name: float(x_value), y_name: float(y_values[iy])}
            try:
                point_result = _reconcile_point(system, pins, parent_seed, pinned_inputs)
            except Exception as exc:  # a failed point never aborts the scan
                records.append({"iy": iy, "success": False, "termination": f"reconcile raised: {exc}"})
                continue
            if not point_result.get("success"):
                records.append({"iy": iy, "success": False, "termination": str(point_result.get("termination"))})
                continue
            parent_seed = dict(system.values)
            records.append({"iy": iy, "success": True, "fields": _point_fields(system, output_names)})
    return records


def _pin_axes_and_compile(system: Any, x_name: str, x_mid: float, y_name: str, y_mid: float) -> dict[str, Any]:
    """Pin both axes (midpoint values) and compile once; return the pinned inputs.

    One compile serves the whole grid: the active relation set depends only on
    *which* variables are supplied, never on their numeric values.
    """
    for name, value in ((x_name, x_mid), (y_name, y_mid)):
        system.inputs[name] = float(value)
        system.values[name] = float(value)
        system.fixed.add(name)
    system.compile()
    return dict(system.inputs)


def _pointwise_chunk_worker(task: tuple) -> list[dict[str, Any]]:
    """Worker: rebuild the system once, then march a chunk of columns.

    Top-level so it pickles for the process pool.  Returns the flat list of
    cell records, each tagged with its ``ix``.
    """
    spec, x_name, y_name, y_values, iy0, columns, output_names = task
    system = _rebuild_system(spec)
    x_mid = float(columns[len(columns) // 2][1])
    pinned_inputs = _pin_axes_and_compile(system, x_name, x_mid, y_name, float(y_values[iy0]))
    records: list[dict[str, Any]] = []
    for ix, x_value, seed in columns:
        for record in _march_column(
            system, pinned_inputs, x_name, x_value, y_name, y_values, iy0, seed, output_names
        ):
            record["ix"] = int(ix)
            records.append(record)
    return records


def run_pointwise(
    system: Any,
    *,
    x: Any = None,
    y: Any = None,
    outputs: Sequence[str] | None = None,
    verbose: int = 0,
    workers: int | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """2-D scan where every point is its own reconciled, verified system.

    Unlike the batched :func:`run`, each grid point is solved by a full
    ``reconcile`` -- so it carries the same guarantee a single scenario does,
    and nonlinear couplings the batched replay cannot close (notably the
    density-peaking cycle ``density_peaking -> n_e -> p_th -> beta_T -> Angioni``)
    are solved properly.

    Points are visited by **continuation** from the reactor's own declared
    operating point: the reference cell first, then its row (serial, each cell
    warm-started from the neighbour just solved), then every column marched up
    and down from its reference-row cell.  Columns are mutually independent, so
    they run on a process pool (``workers``; ``None`` uses the CPU count,
    ``0``/``1`` stays in-process).  Workers rebuild the system from a picklable
    recipe -- live systems cannot cross a process boundary -- and each solves a
    contiguous chunk of columns so the per-worker compile is amortised.

    A point whose reconcile fails does not poison its children: they fall back
    to the nearest solved seed (previous cell in the half-column, else the
    column's reference-row solution).
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

    saved_inputs = dict(self.inputs)
    saved_values = dict(self.values)
    saved_fixed = set(self.fixed)

    nx, ny = int(x_values.size), int(y_values.size)
    # The reference cell is the grid cell closest to the reactor's own declared
    # operating point -- a scenario that already reconciles -- so the very first
    # solve starts from a physical state rather than a cold seed.
    ix0 = int(np.argmin(np.abs(x_values - float(saved_inputs.get(x_name, x_values[nx // 2])))))
    iy0 = int(np.argmin(np.abs(y_values - float(saved_inputs.get(y_name, y_values[ny // 2])))))

    output_names = (
        [self.variable_registry.resolve(str(name)) for name in outputs] if outputs is not None else None
    )
    fields: dict[str, np.ndarray] = {}
    success = np.zeros((ny, nx), dtype=bool)
    failures: list[dict[str, Any]] = []

    def record_point(iy: int, ix: int, record: Mapping[str, Any]) -> None:
        if not record.get("success"):
            failures.append({"ix": int(ix), "iy": int(iy), "termination": str(record.get("termination"))})
            return
        success[iy, ix] = True
        for name, value in record["fields"].items():
            grid = fields.get(name)
            if grid is None:
                grid = fields[name] = np.full((ny, nx), np.nan)
            grid[iy, ix] = value

    try:
        pinned_inputs = _pin_axes_and_compile(
            self, x_name, float(x_values[nx // 2]), y_name, float(y_values[ny // 2])
        )

        # ── Phase A (serial): reference cell, then its row by continuation. ──
        row_solutions: dict[int, dict[str, Any]] = {}
        reference_solution: dict[str, Any] | None = None
        row_order = [ix0, *range(ix0 + 1, nx), *range(ix0 - 1, -1, -1)]
        for ix in row_order:
            neighbour = ix - 1 if ix > ix0 else ix + 1
            seed = row_solutions.get(neighbour) if ix != ix0 else None
            if seed is None:
                seed = reference_solution
            pins = {x_name: float(x_values[ix]), y_name: float(y_values[iy0])}
            try:
                point_result = _reconcile_point(self, pins, seed, pinned_inputs)
            except Exception as exc:
                record_point(iy0, ix, {"success": False, "termination": f"reconcile raised: {exc}"})
                continue
            if not point_result.get("success"):
                record_point(iy0, ix, {"success": False, "termination": str(point_result.get("termination"))})
                continue
            row_solutions[ix] = dict(self.values)
            if reference_solution is None:
                reference_solution = row_solutions[ix]
            record_point(iy0, ix, {"success": True, "fields": _point_fields(self, output_names)})
        if verbose:
            print(f"popcon pointwise: reference row {len(row_solutions)}/{nx} solved")

        # ── Phase B: columns, independent -> process pool (or in-process). ──
        if ny > 1:
            columns = [
                (ix, float(x_values[ix]), row_solutions.get(ix) or reference_solution)
                for ix in range(nx)
            ]
            pool_size = os.cpu_count() or 1 if workers is None else int(workers)
            pool_size = max(1, min(pool_size, nx))
            if pool_size == 1:
                for ix, x_value, seed in columns:
                    for record in _march_column(
                        self, pinned_inputs, x_name, x_value, y_name, y_values, iy0, seed, output_names
                    ):
                        record_point(record["iy"], ix, record)
            else:
                from concurrent.futures import ProcessPoolExecutor

                spec = _system_spec(self)
                chunks = [columns[i::pool_size] for i in range(pool_size)]
                tasks = [
                    (spec, x_name, y_name, y_values, iy0, chunk, output_names)
                    for chunk in chunks
                    if chunk
                ]
                with ProcessPoolExecutor(max_workers=pool_size) as executor:
                    for records in executor.map(_pointwise_chunk_worker, tasks):
                        for record in records:
                            record_point(record["iy"], record["ix"], record)
        if verbose:
            print(f"popcon pointwise: {int(success.sum())}/{nx * ny} points reconciled")
    finally:
        self.inputs.clear()
        self.inputs.update(saved_inputs)
        self.values.clear()
        self.values.update(saved_values)
        self.fixed.clear()
        self.fixed.update(saved_fixed)
        self.initial_guesses.clear()
        self.compile()

    n = nx * ny
    n_ok = int(success.sum())
    result.update(
        {
            "success": n_ok > 0,
            "termination": f"popcon pointwise reconcile: {n_ok}/{n} points solved",
            "n_points": n,
            "n_failed": n - n_ok,
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


def run(
    system: Any,
    *,
    x: Any = None,
    y: Any = None,
    outputs: Sequence[str] | None = None,
    verbose: int = 0,
    solver: str = "batched",
    workers: int | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Batched 2-D scan: evaluate, solve the cores, certify every point.

    ``solver="reconcile"`` instead runs :func:`run_pointwise`, where every grid
    point is its own fully reconciled (and therefore verified) system,
    warm-started by continuation from the reactor's declared operating point,
    with the independent columns solved on a process pool (``workers``).
    Slower, but it closes nonlinear couplings the batched replay cannot.

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
    if str(solver) == "reconcile":
        return run_pointwise(system, x=x, y=y, outputs=outputs, verbose=verbose, workers=workers, **_unused)
    self = system
    result = new_result(self, "popcon")
    if reject_unknown_options(result, _unused):
        return result
    if str(solver) != "batched":
        result["errors"].append(f"popcon solver must be 'batched' or 'reconcile', got {solver!r}.")
        result["termination"] = "invalid options"
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
        _refresh_batched(self, ns, n, trust)
        _solve_cores_batched(self, ns, components, n, trust)

        # Per-point certification through the same machinery as every other
        # mode -- the batched computation only produced candidates.
        fields = {name: np.full((ny, nx), np.nan) for name in output_names}
        success = np.zeros((ny, nx), dtype=bool)
        failures: list[dict[str, Any]] = []
        for index in range(n):
            iy, ix = divmod(index, nx)
            point = slice_point(self, ns, index)
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
