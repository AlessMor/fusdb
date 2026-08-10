"""Reconcile mode.

This module owns the reconciliation execution algorithm. RelationSystem supplies
compiled graph/residual/storage primitives; this mode controls initial computation, global
solve phases, final certification and state mutation policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import time

import numpy as np
from scipy.optimize import least_squares

from . import verify as verify_mode
from .verify import verify_values
from ._common import (
    diagnostics_block,
    new_result,
    record_uninitialized_failure,
    reject_unknown_options,
    result_from_certificate,
    solver_report,
)


def run(
    system: Any,
    *,
    max_nfev: int | None = None,
    movement_weight: float = 1.0,
    irls_iterations: int = 3,
    movement_eps: float = 0.1,
    relation_weight: float = 1.0,
    relation_weight_schedule: Iterable[float] | None = None,
    initial_guesses: Any = None,
    exact: bool = False,
    verbose: int = 0,
    **_unused: Any,
) -> dict[str, Any]:
    """Run structural simultaneous reconciliation.

    The objective changes the fewest non-fixed inputs *beyond their tolerance*.
    Input movement is free within tolerance (a deadzone) and penalised by a
    reweighted-L1 term beyond it; after the relation-weight continuation reaches
    a solution, ``irls_iterations`` iteratively-reweighted-L1 passes re-solve
    with weights ``1/(excess + movement_eps)`` so marginal inputs are pushed back
    inside tolerance and only a sparse set is left changed.  ``movement_eps``
    controls aggressiveness (smaller = sparser, less stable); ``movement_weight``
    scales the movement term.  The solved-state input deviations are returned as
    ``inputs_beyond_tolerance``.

    Two strengths of input anchoring exist, chosen at ingestion:

    * **fixed** values are pinned exactly -- they are never solver unknowns,
      and final verification fails if a solved state contradicts them;
    * **supplied** (non-fixed) inputs are movement-anchored: free inside
      their tolerance band, penalised beyond it as above.

    ``exact=True`` removes the free tolerance band from the *objective*:
    supplied inputs are penalised from the first deviation (in tolerance
    units), so a consistent system reconciles with essentially no input
    movement and an inconsistent one distributes the smallest movement the
    relations allow.  This is only well-posed when the system is square or
    underdetermined -- over-determined inconsistent inputs have no exact
    solution, which is what tolerances exist to absorb.  The
    ``inputs_beyond_tolerance`` report keeps its tolerance-based meaning
    either way.
    """
    self = system
    mode = "reconcile"
    # exact=True drops the movement deadzone in every objective term; the
    # residual, the IRLS weights and the grouped Jacobian must all agree.
    deadzone = not exact
    result = new_result(self, mode)
    if reject_unknown_options(result, _unused):
        return result
    # Initial guesses are precomputed by compile() (run() always compiles
    # before dispatching); reported here so the solver block records how many
    # variables were seeded.  Caller-supplied ``initial_guesses`` (a warm
    # start, e.g. a neighbouring popcon point's solution) are merged on top
    # HERE rather than before run(): the compile re-runs the seeding oracle
    # and would silently wipe anything injected earlier.
    if initial_guesses:
        self.initial_guesses.update(
            {name: value for name, value in dict(initial_guesses).items() if name not in self.fixed and value is not None}
        )
    initial_values: dict[str, Any] = dict(self.initial_guesses)
    # If the current variable state already satisfies the compiled
    # active graph, reconcile is a no-op.  This makes ordered/reconcile
    # adapter blocks idempotent and avoids tiny least-squares drift when the
    # relation set is already certified.
    current_values = self.solver_values()
    current_certificate = verify_values(self, current_values, complete=True)
    if bool(current_certificate.get("verified", False)):
        # Completion may have filled missing outputs even though no numerical
        # solve is needed.  Reconciliation owns that state change: persist the
        # completed namespace just as the solved path below does.
        self.store(current_certificate["values"])
        self.refresh_scales()
        solver = solver_report(
            message="already verified; no reconcile solve",
            residual_size=int(current_certificate["residuals"].size),
            relation_weight=float(relation_weight),
            movement_weight=float(movement_weight),
            initial_guess_variables=int(len(initial_values)),
        )
        return result_from_certificate(
            self,
            mode,
            current_certificate,
            termination="already verified; no reconcile solve",
            solver=solver,
            include_values=True,
            extra={
                "uninitialized_free_variables": [],
                "diagnostics": diagnostics_block(self, verbose=verbose),
            },
        )

    try:
        x0, lower, upper = self.pack()
    except Exception as exc:
        result["errors"].append(str(exc))
        result["termination"] = "initialization failed"
        return result

    if record_uninitialized_failure(self, result):
        return result

    if max_nfev is None:
        # Per-stage budget for one continuation phase.  Trust-region solves on
        # these stiff, highly nonlinear systems (reactivity is near-exponential
        # in temperature) empirically need tens of evaluations per packed
        # dimension to reach the gtol/ftol plateau, so the budget scales with
        # dimension with a floor that lets small well-posed systems finish and a
        # cap that keeps genuinely inconsistent cases from running unbounded.
        # Such cases still stop at the gtol/ftol plateau well before exhausting
        # this budget; callers can pass max_nfev explicitly to tighten or extend.
        max_nfev = int(min(600, max(100, 30 * int(x0.size))))
    if x0.size == 0:
        validation = verify_mode.run(self)
        validation["mode"] = mode
        validation["termination"] = "no free variables; validation only"
        return validation

    profile_spans = [
        {"name": name, "size": int(stop - start)}
        for name, start, stop, _offsets, _scales, shape, _transform in self.packed_specs
        if shape == 1
    ]

    # Movement residuals anchor only immutable input values (frozen into the
    # pack-time movement plan), never initial guesses or relation-completed
    # derived values.
    residual_calls = 0
    residual_eval_time_s = 0.0
    current_relation_weight = float(relation_weight)
    current_movement_weight = float(movement_weight)
    # IRLS movement weights; mode-owned, starting from plain L1 (all ones).
    irls_weights: dict[str, float] = {}
    # Frozen residual-row layout (see RelationSystem.residual_layout),
    # re-frozen per stage by run_stage.  Every evaluation of a stage fills
    # exactly these rows -- a missing value penalizes its own rows -- so the
    # vector size SciPy sees can never drift mid-solve.
    layout: dict[str, Any] = {}
    # First few whole-stage residual failures (S10a): the solver keeps running
    # on the 1e12 barrier, but the original causes survive into diagnostics
    # instead of an unexplained "did not converge".
    residual_failures: list[str] = []

    def residual_function(x: np.ndarray) -> np.ndarray:
        nonlocal residual_calls, residual_eval_time_s
        t0 = time.perf_counter()
        residual_calls += 1
        size = int(layout["size"])
        try:
            values = self.unpack(x)
            # Missing or temporarily invalid relations already contribute large finite residuals.
            blocks = [current_relation_weight * self.layout_relation_rows(values, layout)]
            domain_rows = self.layout_domain_rows(values, layout)
            if domain_rows.size:
                blocks.append(current_relation_weight * domain_rows)
            if layout["movement_names"] is not None:
                blocks.append(current_movement_weight * self.layout_movement_rows(values, layout, irls_weights, deadzone=deadzone))
            parts = [block for block in blocks if block.size]
            out = np.concatenate(parts) if parts else np.empty(0, dtype=float)
            if out.size != size or not np.all(np.isfinite(out)):
                return np.full(size, 1.0e12, dtype=float)
            return out
        except Exception as exc:
            if len(residual_failures) < 5:
                residual_failures.append(f"{type(exc).__name__}: {exc}")
            return np.full(size, 1.0e12, dtype=float)
        finally:
            residual_eval_time_s += time.perf_counter() - t0

    def verify_candidate(x: np.ndarray) -> tuple[dict[str, Any], np.ndarray, list[str], list[str], bool, dict[str, Any], dict[str, Any]]:
        solved = self.unpack(x)
        certificate = verify_values(self, solved, complete=False)
        return (
            certificate["relation_status"],
            certificate["residuals"],
            certificate["errors"],
            certificate["warnings"],
            bool(certificate["verified"]),
            certificate["values"],
            certificate,
        )

    if relation_weight_schedule is None:
        # Escalating relation-weight continuation.  Movement penalties anchor
        # supplied input data in every stage: moving data must always cost
        # something, otherwise degenerate states that zero the whole system
        # satisfy the relation residuals for free.  Seeded variables
        # carry no movement rows, so they stay free to move wherever the
        # relations require.  (A single stage at the high weight was measured
        # 2026-07: it worsens the inputs-beyond-tolerance count on STELLARIS
        # and STEP and slows DEMO, whose warm-up stage verifies in a few
        # evaluations and exits early -- the warm-up is load-bearing.)
        phase_schedule = [(1.0, float(movement_weight)), (100.0, float(movement_weight))]
        weight_schedule = tuple(weight for weight, _move in phase_schedule)
    else:
        weight_schedule = tuple(float(item) for item in relation_weight_schedule)
        if not weight_schedule:
            weight_schedule = (float(relation_weight),)
        # Explicit caller schedule means: use the requested relation weights
        # with the requested movement weight, without an implicit feasibility phase.
        phase_schedule = [(float(weight), float(movement_weight)) for weight in weight_schedule]

    # Freeze an initial layout at x0: this exercises one full
    # unpack/completion (surfacing initialization failures early) and decides
    # the no-residuals early exit.  Stages re-freeze it on their own iterate.
    try:
        layout = self.residual_layout(self.unpack(x0), include_movement=bool(phase_schedule[0][1]))
    except Exception as exc:
        result["errors"].append(f"Residual initialization failed: {exc}")
        result["termination"] = "initialization failed"
        return result
    if int(layout["size"]) == 0:
        validation = verify_mode.run(self)
        validation["mode"] = mode
        validation["termination"] = "no residuals; validation only"
        return validation

    stage_history: list[dict[str, Any]] = []
    solve_result = None
    current_x = np.asarray(x0, dtype=float)
    relation_status: dict[str, dict[str, Any]] = {}
    completed_values: dict[str, Any] = {}
    certificate: dict[str, Any] = {}
    verified = False
    jac_sparsity_used = False
    jac_sparsity = None
    final_probe_size = int(layout["size"])
    try:
        base_common_kwargs = {
            "bounds": (lower, upper),
            "method": "trf",
            "max_nfev": max_nfev,
            # SciPy's default ftol=1e-8 reads a conservative first trust-region
            # step as convergence and quits after one iteration on these stiff
            # systems (e.g. STELLARIS stalled at nfev=2 with the steady-state
            # balances grossly unsatisfied).  Tightening ftol lets the solve keep
            # making progress; it still terminates on gtol/xtol once converged.
            # (2026-07: relaxing to 1e-10/1e-11 and lowering the max_nfev cap
            # were both measured and REJECTED -- ARC/STELLARIS lose
            # inputs-beyond-tolerance quality; the grinding stages stop on the
            # nfev cap or gtol/xtol, not ftol.)
            "ftol": 1.0e-12,
            "verbose": int(verbose),
        }

        def run_stage(stage_label: Any, *, stage_max_nfev: int, include_movement: bool) -> dict[str, Any]:
            """Run one bounded least-squares stage from ``current_x`` and record it.

            Shared by the relation-weight continuation and the IRLS loop.
            Mutates the shared solver state (``current_x``, ``solve_result``,
            the verification products) and appends the stage-history entry,
            which is also returned so a caller may extend it.
            """
            nonlocal solve_result, current_x, relation_status, completed_values, certificate, verified
            nonlocal final_probe_size, jac_sparsity_used, jac_sparsity, layout
            t_stage = time.perf_counter()
            # Stage objectives differ (movement rows come and go with the
            # movement weight), so the row layout is re-frozen per stage on
            # the current iterate; every evaluation of this stage then fills
            # exactly these rows.
            layout = self.residual_layout(self.unpack(current_x), include_movement=include_movement)
            final_probe_size = int(layout["size"])
            stage_kwargs = dict(base_common_kwargs)
            stage_kwargs["max_nfev"] = stage_max_nfev
            # Jacobian strategy, best first.  The structural pattern follows
            # the completion-dependency graph and is conservative: it has an edge
            # for every output of every completion relation, so it never omits a
            # real dependency (including ones that flow through a relation's side
            # outputs, e.g. ARC_V0's n_i/fusion power vs f_He).  Either form is
            # accepted only when its shape exactly matches the frozen layout;
            # any mismatch falls back to scipy's dense differences, so a stale
            # pattern can never make the solve wrong -- only slower.
            stage_jac_mode = "dense"
            expected_shape = (int(layout["size"]), int(current_x.size))
            try:
                plan = self.jacobian_plan(layout)
            except Exception:
                plan = None
            if plan is not None and tuple(plan["sparsity"].shape) == expected_shape:
                stage_kwargs["jac"] = grouped_jacobian(
                    self,
                    plan,
                    layout,
                    lower=lower,
                    upper=upper,
                    relation_weight=current_relation_weight,
                    movement_weight=current_movement_weight,
                    irls_weights=irls_weights,
                    residual_function=residual_function,
                    deadzone=deadzone,
                )
                jac_sparsity = plan["sparsity"]
                jac_sparsity_used = True
                stage_jac_mode = "grouped"
            else:
                try:
                    sparsity = self.build_jac_sparsity(layout)
                except Exception:
                    sparsity = None
                if sparsity is not None and sparsity.shape == expected_shape:
                    stage_kwargs["jac_sparsity"] = sparsity
                    jac_sparsity_used = True
                    jac_sparsity = sparsity
                    stage_jac_mode = "sparsity"
            solve_result = least_squares(residual_function, current_x, **stage_kwargs)
            current_x = np.asarray(solve_result.x, dtype=float)
            relation_status, _residuals, _errors, _warnings, verified, completed_values, certificate = verify_candidate(current_x)
            failed_count = sum(
                1
                for item in relation_status.values()
                if item.get("enforced", True) and not item.get("verified", False)
            )
            entry = {
                "stage": stage_label,
                "relation_weight": float(current_relation_weight),
                "movement_weight": float(current_movement_weight),
                "nfev": int(getattr(solve_result, "nfev", -1)),
                "cost": float(getattr(solve_result, "cost", np.nan)),
                "termination": str(getattr(solve_result, "message", "")),
                "verified": bool(verified),
                "failed_relations": int(failed_count),
                "jac_mode": stage_jac_mode,
                "jac_sparsity_used": stage_jac_mode != "dense",
                "residual_size": int(layout["size"]),
                "elapsed_s": float(time.perf_counter() - t_stage),
            }
            stage_history.append(entry)
            return entry

        for stage_index, (weight, move_weight) in enumerate(phase_schedule):
            current_relation_weight = float(weight)
            current_movement_weight = float(move_weight)
            run_stage(
                int(stage_index),
                stage_max_nfev=max_nfev,
                include_movement=bool(current_movement_weight),
            )
            # Verification is independent of the stage objective.  Stop as soon
            # as all enforced relations are simultaneously satisfied.
            if verified:
                break

        # Iteratively-reweighted-L1 sparsification.  Holding the relation weight
        # at the continuation's final (high) value keeps relations satisfied
        # while the movement L1 is reweighted from the current solution, so each
        # pass pushes inputs that are only marginally past tolerance back inside
        # and leaves a sparse set changed.  Only run once the relations actually
        # verify: on an inconsistent system the inputs cannot be pushed back
        # without breaking a relation, so reweighting cannot lower the count and
        # the extra re-solve would just be wasted (and expensive).
        if verified and movement_weight and int(irls_iterations) > 0:
            current_relation_weight = float(phase_schedule[-1][0])
            current_movement_weight = float(movement_weight)
            # Warm-started reweighting needs only a few steps; cap it so a hard
            # (inconsistent) re-solve cannot burn the full budget per pass.
            irls_max_nfev = int(min(max_nfev, max(40, 15 * int(x0.size))))
            prev_beyond = len(inputs_beyond_tolerance(self, completed_values))
            for irls_index in range(int(irls_iterations)):
                if prev_beyond == 0:
                    break
                # Reweight from the latest solution, then re-solve warm-started.
                irls_weights = self.movement_weights(completed_values, eps=float(movement_eps), deadzone=deadzone)
                entry = run_stage(f"irls_{irls_index}", stage_max_nfev=irls_max_nfev, include_movement=True)
                new_beyond = len(inputs_beyond_tolerance(self, completed_values))
                entry["inputs_beyond_tolerance"] = new_beyond
                # Stop once a pass stops reducing the number beyond tolerance:
                # further reweighting only churns (common for inconsistent data).
                if new_beyond >= prev_beyond:
                    break
                prev_beyond = new_beyond
    except Exception as exc:
        result["errors"].append(f"SciPy least_squares failed: {exc}")
        result["termination"] = "solver error"
        return result
    if solve_result is None:
        result["errors"].append("SciPy least_squares did not run.")
        result["termination"] = "solver error"
        return result

    solver = solver_report(
        backend="scipy.optimize.least_squares",
        success=bool(solve_result.success),
        status=int(getattr(solve_result, "status", 0)),
        cost=float(getattr(solve_result, "cost", np.nan)),
        optimality=float(getattr(solve_result, "optimality", np.nan)),
        nfev=int(getattr(solve_result, "nfev", -1)),
        message=str(solve_result.message),
        residual_calls=int(residual_calls),
        residual_eval_time_s=float(residual_eval_time_s),
        residual_size=int(final_probe_size),
        solver_dim=int(x0.size),
        jac_sparsity_used=bool(jac_sparsity_used),
        jac_sparsity_shape=tuple(jac_sparsity.shape) if jac_sparsity is not None else None,
        residual_eval_mean_ms=float(1000.0 * residual_eval_time_s / max(residual_calls, 1)),
        relation_weight=float(current_relation_weight),
        relation_weight_schedule=[float(item) for item in weight_schedule],
        phase_schedule=[{"relation_weight": float(rw), "movement_weight": float(mw)} for rw, mw in phase_schedule],
        stage_history=stage_history,
        movement_weight=float(current_movement_weight),
        initial_guess_variables=int(len(initial_values)),
        profile_solver_spans=profile_spans,
    )
    validation = result_from_certificate(
        self,
        mode,
        certificate,
        termination=str(solve_result.message),
        solver=solver,
        include_values=True,
        extra={
            "uninitialized_free_variables": list(self.uninitialized_free_variables),
            "diagnostics": diagnostics_block(self, residual_failures=residual_failures, verbose=verbose),
        },
    )

    # There is no separate candidate/final variable state. The latest solve output
    # becomes the current public value and is overwritten on every reconcile call.
    self.store(completed_values)
    self.refresh_scales()
    stored_validation = verify_mode.run(self)
    if bool(stored_validation.get("success", False)) != bool(verified):
        validation["warnings"].append("stored values verify differently after public conversion")
    validation["likely_culprits"] = rank_input_culprits(self, validation.get("relation_status", {}))
    # The objective minimises this set: non-fixed inputs whose solved value left
    # their tolerance band, worst deviation first.
    validation["inputs_beyond_tolerance"] = inputs_beyond_tolerance(self, completed_values)
    return validation


def grouped_jacobian(
    system: Any,
    plan: Mapping[str, Any],
    layout: Mapping[str, Any],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    relation_weight: float,
    movement_weight: float,
    irls_weights: Mapping[str, float],
    residual_function: Any,
    deadzone: bool = True,
):
    """Return the grouped two-point-difference Jacobian callable for one stage.

    Differentiates exactly the stage objective ``residual_function``, one
    perturbation per column group of the finite-difference coloring (``plan``,
    from :meth:`RelationSystem.jacobian_plan`) -- but each group evaluation
    re-runs only the completion providers and relation rows downstream of the
    perturbed columns, while the cheap vectorized domain/movement blocks are
    recomputed whole (unaffected rows difference to exactly zero).  Row sizes
    are pinned by ``layout``, so every block difference is well-defined; an
    unexpected group failure falls back to one full residual call for that
    group, which can make the Jacobian slower but never wrong.
    """
    csc = plan["sparsity"]
    groups = plan["groups"]
    dims = layout["relation_dims"]
    offsets: list[tuple[int, int]] = []
    total = 0
    for rdim in dims:
        offsets.append((total, total + rdim))
        total += rdim
    include_movement = layout["movement_names"] is not None
    rel_step = float(np.sqrt(np.finfo(float).eps))

    def jac(x: np.ndarray, *_args: Any) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        values0 = system.unpack(x)
        blocks0: list[np.ndarray] = []
        for rel, rdim in zip(system._enforced_residual_relations, dims):
            rows, _error = system.enforced_residual_block(rel, values0)
            rows = np.asarray(rows, dtype=float).reshape(-1)
            blocks0.append(rows if rows.size == rdim else np.full(rdim, 1.0e12, dtype=float))
        domain0 = system.layout_domain_rows(values0, layout)
        move0 = (
            system.layout_movement_rows(values0, layout, irls_weights, deadzone=deadzone)
            if include_movement
            else np.empty(0, dtype=float)
        )
        m = int(csc.shape[0])
        sign = np.where(x >= 0.0, 1.0, -1.0)
        h = rel_step * sign * np.maximum(1.0, np.abs(x))
        h = np.where(x + h > upper, -np.abs(h), h)
        h = np.where(x + h < lower, np.abs(h), h)
        jacobian = np.zeros((m, x.size), dtype=float)
        f0_weighted: np.ndarray | None = None
        for group in groups:
            cols = group["cols"]
            x_new = x.copy()
            x_new[cols] = x[cols] + h[cols]
            try:
                ns = dict(values0)
                for name in group["deleted"]:
                    ns.pop(name, None)
                for name, start, stop, offs, scales, shape, transform in group["spans"]:
                    local = x_new[start:stop]
                    actual = offs * np.exp(local) if transform == "log" else offs + scales * local
                    ns[name] = actual.copy() if shape == 1 else float(actual[0])
                system.apply_profile_specs(ns)
                system._apply_completion_providers(ns, plan=group["providers"])
                if any(ns.get(name) is None for name in group["deleted"]):
                    raise ValueError("incremental completion left values missing")
                df = np.zeros(m, dtype=float)
                for index in group["relations"]:
                    rows, _error = system.enforced_residual_block(system._enforced_residual_relations[index], ns)
                    rows = np.asarray(rows, dtype=float).reshape(-1)
                    start_row, stop_row = offsets[index]
                    if rows.size != stop_row - start_row:
                        rows = np.full(stop_row - start_row, 1.0e12, dtype=float)
                    df[start_row:stop_row] = relation_weight * (rows - blocks0[index])
                domain_new = system.layout_domain_rows(ns, layout)
                df[total:total + domain0.size] = relation_weight * (domain_new - domain0)
                if move0.size:
                    move_new = system.layout_movement_rows(ns, layout, irls_weights, deadzone=deadzone)
                    df[total + domain0.size:] = movement_weight * (move_new - move0)
            except Exception:
                if f0_weighted is None:
                    parts = [relation_weight * np.concatenate(blocks0)] if blocks0 else []
                    parts.append(relation_weight * domain0)
                    if move0.size:
                        parts.append(movement_weight * move0)
                    f0_weighted = np.concatenate(parts) if parts else np.empty(0, dtype=float)
                df = residual_function(x_new) - f0_weighted
                if df.size != m:
                    continue
            for j in cols:
                rows_idx = csc.indices[csc.indptr[j]:csc.indptr[j + 1]]
                jacobian[rows_idx, j] = df[rows_idx] / h[j]
        return jacobian

    return jac


def inputs_beyond_tolerance(system: Any, values: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return supplied non-fixed inputs whose solved value left their tolerance.

    Reports, per input variable, the maximum tolerance-normalized deviation
    ``|value - input| / tolerance``; an input is "beyond tolerance" when that
    exceeds one.  The count of these is exactly the quantity the reconcile
    objective tries to minimise, so it is surfaced on the result for the
    caller to inspect.

    Args:
        system: The compiled relation system.
        values: Solver-unit namespace produced by the solve.

    Returns:
        ``{"name", "deviation_tol"}`` entries with deviation > 1, worst first.
    """
    out: list[dict[str, Any]] = []
    for name, ref in system.inputs.items():
        if name in system.fixed or ref is None or values.get(name) is None:
            continue
        # excess is (deviation/tol - 1) clipped at 0, so excess > 0 marks a
        # crossing; deviation in tolerance units is then 1 + excess.
        spec = system.spec_of(name)
        excess = spec.movement_excess(values[name], spec.solver_value(ref, system.profile_size), *system.tols_of(name))
        # The optimizer can stop a few parts per million beyond the exact
        # deadzone boundary.  That is numerical termination noise, not a
        # physically meaningful extra input excursion.  Keep the reporting
        # threshold tiny relative to one tolerance width so real crossings are
        # unaffected while values such as 1.00000057 tol are treated as on the
        # boundary.
        if excess > 1.0e-6:
            out.append({"name": name, "deviation_tol": 1.0 + excess})
    return sorted(out, key=lambda item: -item["deviation_tol"])


def rank_input_culprits(system: Any, relation_status: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for rel in system.relations:
        status = relation_status.get(rel.name, {})
        if status.get("verified", True):
            continue
        for name in rel.variables:
            if name in system.known and name not in system.fixed:
                counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "count": count} for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
