"""Popcon mode: a batched 2-D operating-space scan (POPCON).

The mode sweeps a grid over two scalar axis variables (e.g. average electron
density and temperature) and computes, for every grid point simultaneously, a
consistent operating point with the supplied inputs held exactly as given and
the axis values pinned to the grid coordinates.

Each scheduled chunk is one batched computation: every scalar carries a leading
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

The serial default keeps the whole grid in one chunk.  Explicit workers split
it into vectorized chunks and rebuild the compiled system once per worker.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np

from fusdb.modes._batch import map_chunks, parallel_chunk_size
from fusdb.relationsystem import (
    apply_completion_providers_batched,
    coerce_batched,
    relation_outputs,
    slice_point,
)

from ._common import new_result, reject_unknown_options


# Process-local reusable prepared models. Each worker reconstructs a model at
# most once for an equivalent recipe, then creates ephemeral scenario plans.
_WORKER_MODELS: dict[bytes, Any] = {}


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
        name for name in system.variable_roles if name in system.packed_variables
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
# differences from RelationSystem.apply_completion_providers.

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
    for name, avg_name, shape, fixed_value in system.profile_specs:
        if fixed_value is not None:
            ns[name] = np.broadcast_to(np.asarray(fixed_value, dtype=float), (n, profile_size)).copy()
            continue
        if avg_name is None or ns.get(avg_name) is None:
            continue
        ns[name] = ns[avg_name] * np.asarray(shape, dtype=float)
    # Constant-default stage.
    for name, value in system.constant_defaults_solver.items():
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
            if out_name in system.packed_variables and roles.get(out_name) != "fixed"
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


def _system_spec(system: Any) -> dict[str, Any]:
    """Picklable recipe to rebuild an equivalent system in a worker process.

    Registry relations are represented by name. Runtime-generated source-profile
    relations are already picklable and are carried directly, so there is no
    second reconstruction schema to keep in sync. The common profile-grid size
    is carried separately so an arbitrary source sampling cannot reset worker
    geometry.
    """
    return {
        "name": system.name,
        "profile_size": int(system.profile_size),
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
        "relations": [rel if rel.source_kind == "source_profile" else rel.name for rel in system.model.candidate_primary_relations],
        "constraints": system.model.constraints_spec,
    }


def _rebuild_system(spec: Mapping[str, Any]) -> Any:
    from fusdb.registry import RELATIONS, VARIABLES
    from fusdb.relationsystem import RelationSystem
    from fusdb.variable import Variable

    profile_size = int(spec["profile_size"])
    variables = [
        Variable(
            name,
            value=value,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            fixed=fixed,
            size=profile_size if VARIABLES.get(name).shape == 1 else None,
        )
        for name, value, fixed, rel_tol, abs_tol in spec["variables"]
    ]
    relations = [
        RELATIONS.get(item) if isinstance(item, str) else item
        for item in spec["relations"]
    ]
    return RelationSystem(variables, relations, constraints=spec["constraints"], name=spec["name"])


def _solve_batched_cases(
    system: Any,
    cases: Sequence[tuple[int, float, float]],
    *,
    x_name: str,
    y_name: str,
    output_names: Sequence[str],
    cert_relation_names: Sequence[str],
    cert_vars: set[str],
) -> list[dict[str, Any]]:
    """Vectorize one case chunk, then certify and return each point separately."""
    n = len(cases)
    if not n:
        return []
    by_name = {rel.name: rel for rel in system.relations}
    cert_rels = [by_name[name] for name in cert_relation_names if name in by_name]
    flat_x = np.asarray([case[1] for case in cases], dtype=float)
    flat_y = np.asarray([case[2] for case in cases], dtype=float)
    trust: dict[str, str] = {}
    ns = _batched_base(system, n)
    ns[x_name] = flat_x.reshape(n, 1)
    ns[y_name] = flat_y.reshape(n, 1)
    _refresh_batched(system, ns, n, trust)
    _solve_cores_batched(system, ns, _free_cores(system), n, trust)

    records: list[dict[str, Any]] = []
    for local_index, (index, x_value, y_value) in enumerate(cases):
        point = slice_point(system, ns, local_index)
        system.inputs[x_name] = x_value
        system.inputs[y_name] = y_value
        verified, relation_status, errors = _certify(system, point, cert_rels, cert_vars)
        if not verified:
            failed = [name for name, status in relation_status.items() if not status.get("verified", False)]
            detail = ", ".join(failed[:4]) if failed else "; ".join(errors[:2])
            records.append(
                {
                    "index": index,
                    "success": False,
                    "termination": f"certification failed: {detail}",
                }
            )
            continue
        fields: dict[str, float] = {}
        for name in output_names:
            value = point.get(name)
            if value is None:
                continue
            try:
                public = system.public_value(name, value)
                fields[name] = float(np.asarray(public, dtype=float).reshape(-1)[0])
            except (TypeError, ValueError):
                pass
        records.append({"index": index, "success": True, "fields": fields})
    return records


def _solve_batched_cases_from_spec(
    spec: Mapping[str, Any],
    x_name: str,
    y_name: str,
    output_names: Sequence[str],
    cert_relation_names: Sequence[str],
    cert_vars: set[str],
    cases: tuple[tuple[int, float, float], ...],
) -> list[dict[str, Any]]:
    """Process worker entry point for a vectorized POPCON chunk."""
    cache_key = pickle.dumps(spec, protocol=pickle.HIGHEST_PROTOCOL)
    model = _WORKER_MODELS.get(cache_key)
    if model is None:
        model = _rebuild_system(spec)
        _WORKER_MODELS[cache_key] = model
    plan = model.compile()
    return _solve_batched_cases(
        plan,
        cases,
        x_name=x_name,
        y_name=y_name,
        output_names=output_names,
        cert_relation_names=cert_relation_names,
        cert_vars=cert_vars,
    )


def run(
    system: Any,
    *,
    x: Any = None,
    y: Any = None,
    outputs: Sequence[str] | None = None,
    certification_targets: Sequence[str] | None = None,
    verbose: int = 0,
    workers: int | None = None,
    chunk_size: int | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Chunked 2-D scan: vectorize each chunk and certify every point.

    Args:
        x: X-axis spec (see :func:`parse_axis`).
        y: Y-axis spec.
        outputs: Scalar variables to collect; defaults to every active
            scalar.  Output selection controls only what is returned, never
            whether a point is valid (S6-full).
        certification_targets: Opt into a *narrower* certificate covering only
            these quantities and their connectors (the cone), instead of the
            default full-system certificate.  Changing it changes the success
            mask; the result's ``certificate`` block names the scope
            (``"full"`` by default, ``"cone"`` when set) and lists the exact
            relation set checked.
        verbose: Nonzero prints stage progress.
        workers: Worker processes.  The default keeps the grid in one vectorized
            batch; a value greater than one partitions it automatically.
        chunk_size: Explicit points per vectorized chunk.  With no explicit
            ``workers``, multiple chunks use the available CPUs.

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

    nx, ny = int(x_values.size), int(y_values.size)
    n = nx * ny
    grid_x, grid_y = np.meshgrid(x_values, y_values)  # (ny, nx), rows = y
    flat_x, flat_y = grid_x.reshape(-1), grid_y.reshape(-1)

    # Pin the axes on a fresh scenario plan. Plans are immutable with
    # respect to structural inputs, so POPCON never recompiles/mutates the
    # caller's plan in place.
    scan_inputs = dict(self.inputs)
    scan_fixed = set(self.fixed)
    for name, values in ((x_name, x_values), (y_name, y_values)):
        midpoint = float(values[values.size // 2])
        scan_inputs[name] = midpoint
        scan_fixed.add(name)
    self = self.model.compile(inputs=scan_inputs, fixed=scan_fixed)

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

    # Certification scope is decoupled from output selection (S6-full):
    # by default every enforced relation must hold at a certified point, so
    # which outputs you collect never changes whether a point is valid.
    # ``certification_targets`` opts into the narrower cone of just those
    # quantities (and everything connecting them to the inputs).
    if certification_targets is None:
        cert_rels = [rel for rel in self.relations if rel.enforce]
        cert_vars = set(self.active_variable_names)
        cert_scope = "full"
    else:
        targets = [self.variable_registry.resolve(str(name)) for name in certification_targets]
        cert_rels, cert_vars = certification_cone(self, (*targets, x_name, y_name))
        cert_scope = "cone"
    cert_relation_names = tuple(rel.name for rel in cert_rels)
    cases = tuple(
        (index, float(flat_x[index]), float(flat_y[index]))
        for index in range(n)
    )
    effective_chunk_size = chunk_size
    effective_workers: int | None = 1
    if workers not in (None, 0, 1):
        effective_workers = int(workers)
        if effective_chunk_size is None:
            effective_chunk_size = parallel_chunk_size(n, effective_workers)
    elif effective_chunk_size is not None:
        effective_workers = workers

    if effective_workers == 1:
        worker = partial(
            _solve_batched_cases,
            self,
            x_name=x_name,
            y_name=y_name,
            output_names=tuple(output_names),
            cert_relation_names=cert_relation_names,
            cert_vars=set(cert_vars),
        )
    else:
        worker = partial(
            _solve_batched_cases_from_spec,
            _system_spec(self),
            x_name,
            y_name,
            tuple(output_names),
            cert_relation_names,
            set(cert_vars),
        )
    if verbose:
        batches = (n + (effective_chunk_size or n) - 1) // (effective_chunk_size or n)
        print(f"popcon: {n} points in {batches} vectorized chunk(s)")
    records = map_chunks(
        cases,
        worker,
        workers=effective_workers,
        chunk_size=effective_chunk_size,
    )

    fields = {name: np.full((ny, nx), np.nan) for name in output_names}
    success = np.zeros((ny, nx), dtype=bool)
    failures: list[dict[str, Any]] = []
    for record in records:
        index = int(record["index"])
        iy, ix = divmod(index, nx)
        success[iy, ix] = bool(record["success"])
        if record["success"]:
            for name, value in record["fields"].items():
                fields[name][iy, ix] = value
        else:
            failures.append({"ix": int(ix), "iy": int(iy), "termination": record["termination"]})
    if verbose:
        print(f"popcon: {n - len(failures)}/{n} points certified")

    n_failed = len(failures)
    result.update(
        {
            # Infeasible regions are normal in a popcon; the scan succeeds
            # when it ran and at least one operating point certified.
            "success": n_failed < n,
            "termination": f"popcon scan completed: {n - n_failed}/{n} points solved",
            "n_points": n,
            "n_failed": n_failed,
            # Certificate scope (S6-full): ``full`` certifies every enforced
            # relation (the default -- same guarantee as reconcile/verify);
            # ``cone`` certifies only the requested certification_targets and
            # their connectors, leaving variables outside the cone as
            # unconstrained scenario freedom.  The exact checked set is reported.
            "certificate": {
                "scope": cert_scope,
                "checked_relations": sorted(rel.name for rel in cert_rels if rel.enforce),
                "expected_relations": int(sum(1 for rel in self.relations if rel.enforce)),
            },
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
