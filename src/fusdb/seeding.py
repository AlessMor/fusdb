"""Seeding oracle: initial solver values propagated from supplied inputs.

Module-level functions taking the compiled :class:`~fusdb.relationsystem.RelationSystem`
as their first argument (the same convention as :mod:`fusdb.modes`).  They read
the compiled products (active relations, structural blocks, variable roles) and
the promoted per-variable helpers; nothing here writes ``Variable`` state --
the oracle fills a working solver-unit namespace only.

Two entry points are consumed outside this module:

* :func:`initial_values_from_graph` -- the x0 oracle behind
  ``RelationSystem.initial_values_from_graph`` (used by compile-time pruning
  and reported by reconcile).
* :func:`solve_block` -- the small-block numeric solver, shared by the seeding
  fixed point here and by ordered mode's simultaneous blocks (which opt in to
  profile-valued cores via ``allow_profile_core``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from .relation import Relation, is_default_relation
from .utils import signed_scalar_grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .relationsystem import RelationSystem


def initial_values_from_graph(system: "RelationSystem") -> dict[str, Any]:
    """Fill solver start values by direct propagation from supplied values.

    Iteratively solves every relation that has exactly one missing variable
    (the 1x1 / acausal step), to a fixed point.  These are exact values, not
    movement references.  Variables that remain missing are the free unknowns
    of larger coupled blocks (block cores); they are packed directly and
    determined by the simultaneous reconcile against their block's supplied
    anchor, so no separate block solver is needed here.

    Returns:
        ``{name: value}`` for every variable the oracle seeded (supplied
        values are never included).
    """
    values = system.input_values()
    system.apply_profile_specs(values)
    original = set(values)
    seeded: set[str] = set()
    # Constant defaults are known values from the start (they are held, not
    # solved), so downstream propagation can use them.
    for name, value in system.constant_default_values.items():
        if values.get(name) is None:
            try:
                values[name] = system.solver_value(name, value)
                seeded.add(name)
            except Exception:
                pass
    # Propagate everything derivable from the supplied values.
    _propagate_known(system, values, seeded, original)
    # Seed registry defaults for variables that supplied-propagation left
    # missing, then re-propagate so downstream values (n_X = n_i * f_X, ...)
    # fill in.  Defaults are pure x0 seeds -- never enforced -- applied to a
    # fixpoint so variable-reference defaults (T_i = T_e) resolve once their
    # source has a value.
    for _ in range(50):
        if not _seed_defaults(system, values, seeded, original):
            break
        _propagate_known(system, values, seeded, original)
    return {name: values[name] for name in values if name in seeded}


def _propagate_known(system: "RelationSystem", values: dict[str, Any], seeded: set[str], original: set[str]) -> None:
    """Fill values derivable from the currently known namespace.

    Stage 1 runs direct 1x1/acausal propagation to a fixed point; stage 2
    solves the determined blocks (2x2 ... N x N) for their cores, with a
    final merged-block sweep for variables left in no individual block.
    """
    # Stage 1: direct 1x1/acausal propagation to a fixed point.
    for _direct_pass in range(50):
        if not _compute_direct_outputs(system, values, seeded, original):
            break
    # Stage 2: solve the determined blocks (2x2 ... N x N) for their cores.
    progress = True
    while progress:
        progress = False
        for block in system.structural_blocks:
            if _compute_planned_block(system, block, values, seeded, original):
                progress = True
                for _direct_pass in range(50):
                    if not _compute_direct_outputs(system, values, seeded, original):
                        break
    merged = tuple(
        name
        for block in system.structural_blocks
        for name in block
        if (name not in values or values[name] is None) and name not in original
    )
    if merged and _compute_planned_block(system, merged, values, seeded, original):
        for _direct_pass in range(50):
            if not _compute_direct_outputs(system, values, seeded, original):
                break


def _seed_defaults(system: "RelationSystem", values: dict[str, Any], seeded: set[str], original: set[str]) -> bool:
    """Seed still-missing active variables from their registry default.

    A default is either a number (a constant x0 seed) or the name of another
    variable (copy that variable's current value).  Seeds are pure initial
    points: a variable a relation determines is moved off its seed by the
    global solve, and a variable no enforced relation touches keeps it
    (zero-gradient).  A default is applied only when the variable is active,
    not supplied/fixed and still missing; variable-reference defaults whose
    source is not yet known are skipped (the caller iterates to a fixpoint).
    """
    progress = False
    for name, role in sorted(system.variable_roles.items()):
        if role == "inactive":
            continue
        if name in values and values[name] is not None:
            continue
        if name in original or name not in system.variable_registry:
            continue
        if name in system.fixed or system.inputs.get(name) is not None:
            continue
        spec = system.variable_registry.get(name)
        default = spec.default
        if default is None:
            continue
        if spec.default_requires is not None:
            required = system.variable_registry.resolve(spec.default_requires)
            if values.get(required) is None:
                continue
        if isinstance(default, str):
            if default not in system.variable_registry:
                continue
            source = system.variable_registry.get(default).canonical_name
            if source not in values or values[source] is None:
                continue
            raw: Any = values[source]
        else:
            raw = float(default)
        try:
            value = system.solver_value(name, raw)
            if not system.spec_of(name).candidate_valid(value, system.profile_size):
                continue
        except Exception:
            continue
        values[name] = value
        seeded.add(name)
        progress = True
    return progress


def _direct_relation_pool(system: "RelationSystem") -> list[Relation]:
    """Relations allowed for direct output initial computation.

    The global reconcile still uses ``system.relations``.  For initial guesses
    only, inactive weak/default providers may fill missing values when their
    inputs are already known.  This makes defaults useful as x0 generators
    without adding them as extra enforced residuals or movement references.
    """
    by_name = {rel.name: rel for rel in system.relations}
    for rel in system.candidate_primary_relations:
        if rel.name in by_name:
            continue
        if is_default_relation(rel):
            by_name[rel.name] = rel
    return list(by_name.values())


def _seed_accepts(system: "RelationSystem", name: str, original: set[str]) -> bool:
    """Return whether seeding may write a value for one variable.

    Seeding only fills genuinely missing degrees of freedom: it never
    overrides a user-supplied value (``original``) or a fixed variable, and
    it ignores names the registry does not know.

    Args:
        system: The compiled relation system.
        name: Candidate variable name to write.
        original: Names that already had a value before seeding began.

    Returns:
        ``True`` when ``name`` may be written by seeding.
    """
    if name not in system.variable_registry or name in original:
        return False
    return name not in system.fixed


def _compute_direct_outputs(system: "RelationSystem", values: dict[str, Any], seeded: set[str], original: set[str]) -> bool:
    """Seed values by solving every relation that has exactly one unknown.

    Seeding is *adirectional*: a relation is an equation, so whenever all but
    one of its variables are known the remaining one is obtained by inverting
    the canonical relation through :meth:`Relation.solve`, regardless of
    whether that variable is a declared input or a declared output.  This is
    the single-unknown step the seeding fixed point is built on; the caller
    repeats it until nothing more can be solved, then hands the remaining
    coupled variables to the block solver.

    A relation whose inputs are all known but which still has several unknown
    outputs is additionally evaluated forward to fill those outputs together
    (each is an independent forward computation, not a directional
    assumption).  Implicit relations are skipped: a variable appearing on
    both sides cannot be trusted to a blind inverse.  Only missing, non-fixed,
    non-supplied variables are written, and every value must be finite and
    in-domain, so ``seeded`` stays a pure record of seeded names.

    Args:
        system: The compiled relation system.
        values: Working solver-unit namespace, mutated in place.
        seeded: Names seeded so far, mutated in place.
        original: Names supplied before seeding (never overwritten).

    Returns:
        ``True`` if any value was filled this pass.
    """
    progress = False
    for rel in _direct_relation_pool(system):
        # Primary path: a relation with exactly one unknown variable is
        # solved in whatever direction closes it (input or output).
        if not rel.implicit:
            unknown = [name for name in rel.variables if values.get(name) is None]
            if len(unknown) == 1 and _seed_accepts(system, unknown[0], original):
                name = unknown[0]
                known = {vname: values[vname] for vname in rel.variables if vname != name}
                try:
                    value = system.solver_value(name, rel(**known))
                    if system.spec_of(name).candidate_valid(value, system.profile_size):
                        values[name] = value
                        seeded.add(name)
                        progress = True
                        continue
                except Exception:
                    pass

        # Secondary path: every input is known, so any still-missing outputs
        # are each computable forward in one evaluation.
        if rel.output_names and all(not values.get(inp) is None for inp in rel.input_names):
            try:
                mapped = rel.output_map(rel.evaluate(system.relation_evaluation_values(rel, values)))
            except Exception:
                mapped = {}
            for name in rel.output_names:
                if name not in mapped or not values.get(name) is None or not _seed_accepts(system, name, original):
                    continue
                try:
                    value = system.solver_value(name, mapped[name])
                    if not system.spec_of(name).candidate_valid(value, system.profile_size):
                        continue
                except Exception:
                    continue
                values[name] = value
                seeded.add(name)
                progress = True
    return progress


def _compute_planned_block(
    system: "RelationSystem",
    block: tuple[str, ...],
    values: dict[str, Any],
    seeded: set[str],
    original: set[str],
) -> bool:
    """Solve one structurally planned block as initial guesses.

    The block is first extended with every variable producible from it
    (generated profiles, reactivities, rates), so relations that pin the
    block only through those intermediates participate in the solve.
    """
    unknowns = tuple(name for name in block if name not in values or values[name] is None)
    if not unknowns or any(name in original for name in unknowns):
        return False
    for name in unknowns:
        if name not in system.known or name in system.fixed:
            return False
    extended, rels = _block_closure(system, unknowns, values)
    if not rels:
        return False
    solved = solve_block(system, extended, rels, values, residual_tol=1.0)
    if solved is None:
        return False
    for name, value in solved.items():
        values[name] = value
        seeded.add(name)
    return True


def _block_closure(system: "RelationSystem", unknowns: tuple[str, ...], values: Mapping[str, Any]) -> tuple[tuple[str, ...], list[Relation]]:
    """Extend a planned block with variables producible from it.

    Returns the extended unknown set and the participating relations:
    every active relation whose missing variables are covered by the
    extended set.  Non-enforced relations participate as value providers
    only; enforced relations supply the residual rows.
    """
    available = {name for name, value in values.items() if value is not None}
    extended = set(unknowns)
    changed = True
    while changed:
        changed = False
        for rel in system.relations:
            if rel.implicit or not rel.output_names:
                continue
            if not all(inp in available or inp in extended for inp in rel.input_names):
                continue
            for out in rel.output_names:
                if out in available or out in extended or out not in system.variable_registry:
                    continue
                if out in system.fixed:
                    continue
                extended.add(out)
                changed = True
    rels: list[Relation] = []
    for rel in system.relations:
        missing = {name for name in rel.variables if name not in available}
        if not missing or not missing <= extended:
            continue
        rels.append(rel)
    return tuple(sorted(extended)), rels


def solve_block(
    system: "RelationSystem",
    unknowns: tuple[str, ...],
    rels: list[Relation],
    values: Mapping[str, Any],
    *,
    residual_tol: float,
    allow_profile_core: bool = False,
) -> dict[str, Any] | None:
    """Solve one small initial-computation block; returns ``{name: value}`` or None.

    Unknowns that are declared outputs of a block relation are recomputed
    from that relation inside the residual, so the numerical search runs
    only over the remaining core unknowns.  Core starts come from
    supplied/current values when present, otherwise from the magnitudes
    of the known values around the block, refined by a coarse log-grid
    scan.  Solver domains constrain the search; they never provide a
    start on their own.

    The reconcile seeding path keeps the numerical core scalar so the search
    dimension never explodes pointwise (profiles are recomputed from their
    producers).  Ordered-mode blocks pass ``allow_profile_core=True``: a
    profile-valued core is then packed per element (log for positive-bounded
    elements, affine otherwise) and the coarse grid refinement is applied to
    the scalar core members only.
    """
    for name in unknowns:
        if name not in system.known or name in system.fixed:
            return None

    producers = _block_producers(system, unknowns, rels, values)
    core = [name for name in unknowns if name not in producers]
    if not core:
        # A fully produced cycle still needs one numerical degree of freedom.
        core = [unknowns[0]]
        producers.pop(unknowns[0], None)
    profile_core = {name for name in core if system.spec_of(name).shape == 1}
    if profile_core and not allow_profile_core:
        return None
    scalar_core = [name for name in core if name not in profile_core]
    core_dim = len(scalar_core) + len(profile_core) * system.profile_size
    enforced_rows = sum(max(1, system.relation_row_dim(rel)) for rel in rels if rel.enforce)
    if enforced_rows < core_dim:
        return None

    def namespace_from(core_values: Mapping[str, Any]) -> dict[str, Any]:
        ns = dict(values)
        ns.update(core_values)
        for produced, rel in producers.items():
            try:
                mapped = rel.output_map(rel.evaluate(system.relation_evaluation_values(rel, ns)))
                if mapped.get(produced) is not None:
                    ns[produced] = system.solver_value(produced, mapped[produced])
            except Exception:
                continue
        return system.complete(ns)

    def residual_from(core_values: Mapping[str, Any]) -> np.ndarray:
        ns = namespace_from(core_values)
        blocks: list[np.ndarray] = []
        for rel in rels:
            if not rel.enforce:
                continue
            if any(name not in ns or ns[name] is None for name in rel.variables):
                blocks.append(np.asarray([1.0e6], dtype=float))
                continue
            try:
                eval_values = system.relation_evaluation_values(rel, ns)
                blocks.append(system.relation_residual_vector(rel, eval_values, safe=True))
            except Exception:
                blocks.append(np.asarray([1.0e6], dtype=float))
        out = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        return np.nan_to_num(out, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

    def score(core_values: Mapping[str, Any]) -> float:
        residual = residual_from(core_values)
        return float(np.max(np.abs(residual))) if residual.size else np.inf

    bounds_by_name: dict[str, tuple[float, float]] = {}
    starts: dict[str, Any] = {}
    for name in core:
        lb, ub = system.spec_of(name).solver_bounds
        bounds_by_name[name] = (lb, ub)
        size = system.profile_size if name in profile_core else 1
        elements = np.empty(size, dtype=float)
        for i in range(size):
            try:
                elements[i] = float(system.initial_value(name, index=i if name in profile_core else None))
            except Exception:
                start = _block_start_from_knowns(rels, values, lb, ub)
                if start is None:
                    return None
                elements[i] = start
        starts[name] = elements if name in profile_core else float(elements[0])

    # Coordinate-wise log-grid refinement of the scalar starts.  One sweep is
    # exact for a single core unknown; two sweeps untangle coupled cores.
    # A warm start already in the solution basin (e.g. the neighbouring popcon
    # grid point, fed in through ``initial_guesses``) skips the sweeps: each
    # score evaluation runs the full completion chain, so the sweep dominates
    # the block cost, and a determined block converges to the same unique
    # answer from any in-basin start.  100 tolerance-widths is far inside the
    # basin while a cold or missing start scores ~1e6.
    if score(starts) > 100.0:
        for _sweep in range(1 if len(scalar_core) == 1 else 2):
            for name in scalar_core:
                lb, ub = bounds_by_name[name]
                best, best_score = starts[name], score(starts)
                for point in signed_scalar_grid(lb, ub, decades=30, step=2):
                    point_score = score({**starts, name: point})
                    if point_score < best_score:
                        best, best_score = point, point_score
                starts[name] = best

    # An unconstrained core direction means the block residual does not
    # determine the value: accepting it would seed an arbitrary number.
    # The direction is flat when widely separated grid points give the
    # same in-tolerance score; weak but nonzero dependence is kept.
    for name in scalar_core:
        lb, ub = bounds_by_name[name]
        grid = signed_scalar_grid(lb, ub, decades=30, step=2)
        if len(grid) < 3:
            continue
        probes = [score({**starts, name: point}) for point in (grid[0], grid[len(grid) // 2], grid[-1])]
        if max(probes) - min(probes) <= 1e-9 and min(probes) <= residual_tol:
            return None

    offsets: list[float] = []
    scales: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    transforms: list[str] = []
    spans: list[tuple[str, int, int]] = []
    for name in core:
        lb, ub = bounds_by_name[name]
        elements = np.asarray(starts[name], dtype=float).reshape(-1)
        start_index = len(offsets)
        for element in elements:
            init = min(max(float(element), lb), ub) if np.isfinite(lb) or np.isfinite(ub) else float(element)
            scale, offset, lo, hi, transform = system.pack_scalar(name, init, lb, ub, scale_ref=init)
            offsets.append(offset)
            scales.append(scale)
            lower.append(lo)
            upper.append(hi)
            transforms.append(transform)
        spans.append((name, start_index, len(offsets)))

    def core_values_from(x: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(x, dtype=float)
        out: dict[str, Any] = {}
        for name, start, stop in spans:
            elements = np.empty(stop - start, dtype=float)
            for j, idx in enumerate(range(start, stop)):
                if transforms[idx] == "log":
                    elements[j] = offsets[idx] * np.exp(arr[idx])
                else:
                    elements[j] = offsets[idx] + scales[idx] * arr[idx]
            out[name] = elements if name in profile_core else float(elements[0])
        return out

    def residual(x: np.ndarray) -> np.ndarray:
        return residual_from(core_values_from(x))

    x0 = np.zeros(core_dim, dtype=float)
    try:
        probe = residual(x0)
        if probe.size < core_dim:
            return None
        sol = least_squares(
            residual,
            x0,
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            method="trf",
            x_scale=np.ones_like(x0),
            max_nfev=200 if profile_core else 80,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
    except Exception:
        return None
    final_residual = residual(sol.x)
    max_abs = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
    if not np.isfinite(max_abs) or max_abs > residual_tol:
        return None
    ns = namespace_from(core_values_from(sol.x))
    solved: dict[str, Any] = {}
    for name in unknowns:
        if name not in ns or ns[name] is None:
            return None
        value = ns[name]
        if not system.spec_of(name).candidate_valid(value, system.profile_size):
            return None
        solved[name] = system.solver_value(name, value)
    return solved


def _block_producers(system: "RelationSystem", unknowns: tuple[str, ...], rels: list[Relation], values: Mapping[str, Any]) -> dict[str, Relation]:
    """Return produced-unknown -> relation, in evaluation order.

    A block unknown is produced when one block relation declares it as an
    output and every input of that relation is either known or another
    block unknown.  Producible unknowns that cannot be ordered, because
    they form a cyclic chain, stay in the numerical core.
    """
    unknown_set = set(unknowns)
    available = {name for name, value in values.items() if value is not None}
    producible: set[str] = set()
    for rel in rels:
        if rel.implicit or not rel.output_names:
            continue
        producible.update(out for out in rel.output_names if out in unknown_set)
    # Unknowns no relation can produce are the numerical core seeds.
    available.update(name for name in unknown_set if name not in producible)

    # Greedy topological selection: an unknown is assigned the first
    # relation whose inputs are already available, so purely cyclic
    # alternatives (for example quasineutrality pairs) never deadlock the
    # ordering.  Defaults come first: they carry the weak-assumption
    # value, while enforced alternatives stay in the residual rows.
    ordered_rels = sorted(rels, key=lambda rel: not is_default_relation(rel))
    ordered: dict[str, Relation] = {}
    changed = True
    while changed:
        changed = False
        for rel in ordered_rels:
            if rel.implicit or not rel.output_names:
                continue
            if not all(inp in available for inp in rel.input_names):
                continue
            for out in rel.output_names:
                if out in unknown_set and out not in available:
                    ordered[out] = rel
                    available.add(out)
                    changed = True
    return ordered


def _block_start_from_knowns(rels: list[Relation], values: Mapping[str, Any], lb: float, ub: float) -> float | None:
    """Return a start from the magnitudes of the block's known values.

    This mirrors the standalone relation inverse-solve heuristic: the
    geometric mean of the finite positive known values touching the block,
    clipped into the solver interval.  Bounds only clip; they never
    generate the start themselves.
    """
    magnitudes: list[float] = []
    for rel in rels:
        for name in rel.variables:
            value = values.get(name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float).reshape(-1)
            magnitudes.extend(float(item) for item in arr if np.isfinite(item) and item > 0.0)
    if not magnitudes:
        return None
    start = float(np.exp(np.mean(np.log(np.asarray(magnitudes, dtype=float)))))
    if np.isfinite(lb) and start < lb:
        start = float(lb)
    if np.isfinite(ub) and start > ub:
        start = float(ub)
    return start
