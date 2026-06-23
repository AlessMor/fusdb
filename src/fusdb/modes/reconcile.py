"""Reconcile mode.

This module owns the reconciliation execution algorithm. RelationSystem supplies
compiled graph/residual/storage primitives; this mode controls initial computation, global
solve phases, final certification and state mutation policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import time

import numpy as np
from scipy.optimize import least_squares

from . import verify as verify_mode
from .verify import verify_values


def run(
    system: Any,
    *,
    max_nfev: int | None = None,
    movement_weight: float = 1.0,
    irls_iterations: int = 3,
    movement_eps: float = 0.1,
    relation_weight: float = 1.0,
    relation_weight_schedule: Iterable[float] | None = None,
    verbose: int = 0,
    compute_missing: bool = True,
    initial_residual_tol: float = 1.0,
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
    """
    self = system
    mode = "reconcile"
    # IRLS movement weights; reset so each reconcile starts from plain L1.
    self._movement_weights = {}
    result = self._new_result(mode)
    if self._reject_unknown_options(result, _unused):
        return result
    initial_values: dict[str, Any] = {}
    if compute_missing:
        if getattr(self, "_runtime_initial_values_ready", False):
            initial_values = dict(getattr(self, "_initial_guesses", {}))
        else:
            try:
                initial_values, _initial_info = self._initial_values_from_graph(
                    residual_tol=float(initial_residual_tol),
                )
                self._initial_guesses = dict(initial_values)
            except Exception as exc:
                result["warnings"].append(f"initial graph computation failed: {exc}")
                initial_values = {}
    # If the current variable state already satisfies the compiled
    # active graph, reconcile is a no-op.  This makes ordered/reconcile
    # adapter blocks idempotent and avoids tiny least-squares drift when the
    # relation set is already certified.
    current_values = self._values_from_variables(
        for_solver=True,
        skip_missing=True,
        complete=False,
    )
    current_certificate = verify_values(self, current_values, complete=True)
    if bool(current_certificate.get("verified", False)):
        solver = self._solver_report(
            message="already verified; no reconcile solve",
            residual_size=int(current_certificate["residuals"].size),
            relation_weight=float(relation_weight),
            movement_weight=float(movement_weight),
            initial_guess_variables=int(len(initial_values)),
        )
        return self._result_from_certificate(
            mode,
            current_certificate,
            termination="already verified; no reconcile solve",
            solver=solver,
            include_values=True,
            extra={"uninitialized_free_variables": []},
        )

    try:
        x0, lower, upper, x_scale, spans = self._pack_free_variables()
    except Exception as exc:
        result["errors"].append(str(exc))
        result["termination"] = "initialization failed"
        return result

    if self._record_uninitialized_failure(result):
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

    self._prepare_runtime(spans)
    profile_spans = [
        {"name": span.name, "size": int(span.stop - span.start)}
        for span in spans
        if self.variables_by_name[span.name].shape == 1
    ]

    # Movement residuals anchor only immutable input values, not initial
    # guesses or relation-completed derived values.
    reference = self._values_from_variables(for_solver=True, skip_missing=True, complete=False, use_input_values=True)
    residual_size = 0
    residual_calls = 0
    residual_eval_time_s = 0.0
    current_relation_weight = float(relation_weight)
    current_movement_weight = float(movement_weight)

    def residual_function(x: np.ndarray) -> np.ndarray:
        nonlocal residual_size, residual_calls, residual_eval_time_s, current_relation_weight, current_movement_weight
        t0 = time.perf_counter()
        residual_calls += 1
        try:
            values = self._values_from_vector(x)
            _status, relation_residuals, errors, _warnings = self._evaluate_relation_residuals(
                values,
                strict=True,
                solver_residuals=True,
            )
            # Missing or temporarily invalid relations already contribute large finite residuals.
            blocks = [current_relation_weight * relation_residuals]
            domain_residuals = self._domain_residuals(values)
            if domain_residuals.size:
                blocks.append(current_relation_weight * domain_residuals)
            if current_movement_weight:
                blocks.append(current_movement_weight * self._movement_residuals(values, reference, spans))
            out = np.concatenate([block.reshape(-1) for block in blocks if block.size])
            if not np.all(np.isfinite(out)):
                raise ValueError("non-finite residual vector")
            residual_size = int(out.size)
            return out
        except Exception:
            if residual_size:
                return np.full(residual_size, 1.0e12, dtype=float)
            raise
        finally:
            residual_eval_time_s += time.perf_counter() - t0

    def verify_candidate(x: np.ndarray) -> tuple[dict[str, Any], np.ndarray, list[str], list[str], bool, dict[str, Any], dict[str, Any]]:
        solved = self._values_from_vector(x)
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

    try:
        probe = residual_function(x0)
        residual_size = int(probe.size)
    except Exception as exc:
        result["errors"].append(f"Residual initialization failed: {exc}")
        result["termination"] = "initialization failed"
        return result
    if probe.size == 0:
        validation = verify_mode.run(self)
        validation["mode"] = mode
        validation["termination"] = "no residuals; validation only"
        return validation

    if relation_weight_schedule is None:
        # Escalating relation-weight continuation.  Movement penalties anchor
        # supplied input data in every stage: moving data must always cost
        # something, otherwise degenerate states that zero the whole system
        # satisfy the relation residuals for free.  Seeded variables
        # carry no movement rows, so they stay free to move wherever the
        # relations require.
        phase_schedule = [(1.0, float(movement_weight)), (100.0, float(movement_weight))]
        weight_schedule = tuple(weight for weight, _move in phase_schedule)
    else:
        weight_schedule = tuple(float(item) for item in relation_weight_schedule)
        if not weight_schedule:
            weight_schedule = (float(relation_weight),)
        # Explicit caller schedule means: use the requested relation weights
        # with the requested movement weight, without an implicit feasibility phase.
        phase_schedule = [(float(weight), float(movement_weight)) for weight in weight_schedule]

    stage_history: list[dict[str, Any]] = []
    solve_result = None
    current_x = np.asarray(x0, dtype=float)
    relation_status: dict[str, dict[str, Any]] = {}
    completed_values: dict[str, Any] = {}
    certificate: dict[str, Any] = {}
    verified = False
    jac_sparsity_used = False
    jac_sparsity = None
    final_probe_size = int(probe.size)
    try:
        base_common_kwargs = {
            "bounds": (lower, upper),
            "x_scale": x_scale,
            "method": "trf",
            "max_nfev": max_nfev,
            # SciPy's default ftol=1e-8 reads a conservative first trust-region
            # step as convergence and quits after one iteration on these stiff
            # systems (e.g. STELLARIS stalled at nfev=2 with the steady-state
            # balances grossly unsatisfied).  Tightening ftol lets the solve keep
            # making progress; it still terminates on gtol/xtol once converged.
            "ftol": 1.0e-12,
            "verbose": int(verbose),
        }
        for stage_index, (weight, move_weight) in enumerate(phase_schedule):
            current_relation_weight = float(weight)
            current_movement_weight = float(move_weight)
            t_stage = time.perf_counter()
            # Residual size changes between feasibility-only and movement-regularized
            # stages, so sparsity must be built per stage.
            stage_probe = residual_function(current_x)
            final_probe_size = int(stage_probe.size)
            stage_kwargs = dict(base_common_kwargs)
            # Sparse finite-difference Jacobian.  The structural pattern follows
            # the completion-dependency graph and is conservative: it has an edge
            # for every output of every completion relation, so it never omits a
            # real dependency (including ones that flow through a relation's side
            # outputs, e.g. ARC_V0's n_i/fusion power vs f_He).  It is accepted
            # only when its shape exactly matches the live residual/variable sizes
            # for this stage; any mismatch falls back to dense differences, so a
            # stale pattern can never make the solve wrong -- only slower.
            stage_jac_used = False
            try:
                reference_map = reference if current_movement_weight else None
                sparsity = self._build_jac_sparsity(spans, reference=reference_map)
            except Exception:
                sparsity = None
            if sparsity is not None and sparsity.shape == (int(stage_probe.size), int(current_x.size)):
                stage_kwargs["jac_sparsity"] = sparsity
                stage_jac_used = True
                jac_sparsity_used = True
                jac_sparsity = sparsity
            solve_result = least_squares(residual_function, current_x, **stage_kwargs)
            current_x = np.asarray(solve_result.x, dtype=float)
            relation_status, _residuals, _errors, _warnings, verified, completed_values, certificate = verify_candidate(current_x)
            failed_count = sum(
                1
                for item in relation_status.values()
                if item.get("enforced", True) and not item.get("verified", False)
            )
            stage_history.append(
                {
                    "stage": int(stage_index),
                    "relation_weight": float(weight),
                    "movement_weight": float(move_weight),
                    "nfev": int(getattr(solve_result, "nfev", -1)),
                    "cost": float(getattr(solve_result, "cost", np.nan)),
                    "termination": str(getattr(solve_result, "message", "")),
                    "verified": bool(verified),
                    "failed_relations": int(failed_count),
                    "jac_sparsity_used": bool(stage_jac_used),
                    "residual_size": int(stage_probe.size),
                    "elapsed_s": float(time.perf_counter() - t_stage),
                }
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
            prev_beyond = len(self._inputs_beyond_tolerance(completed_values))
            for irls_index in range(int(irls_iterations)):
                if prev_beyond == 0:
                    break
                # Reweight from the latest solution, then re-solve warm-started.
                self._update_movement_weights(completed_values, reference, spans, eps=float(movement_eps))
                t_stage = time.perf_counter()
                stage_probe = residual_function(current_x)
                final_probe_size = int(stage_probe.size)
                stage_kwargs = dict(base_common_kwargs)
                stage_kwargs["max_nfev"] = irls_max_nfev
                # Conservative structural sparsity here too (see the stage loop above).
                stage_jac_used = False
                try:
                    sparsity = self._build_jac_sparsity(spans, reference=reference)
                except Exception:
                    sparsity = None
                if sparsity is not None and sparsity.shape == (int(stage_probe.size), int(current_x.size)):
                    stage_kwargs["jac_sparsity"] = sparsity
                    stage_jac_used = True
                    jac_sparsity_used = True
                    jac_sparsity = sparsity
                solve_result = least_squares(residual_function, current_x, **stage_kwargs)
                current_x = np.asarray(solve_result.x, dtype=float)
                relation_status, _residuals, _errors, _warnings, verified, completed_values, certificate = verify_candidate(current_x)
                failed_count = sum(
                    1
                    for item in relation_status.values()
                    if item.get("enforced", True) and not item.get("verified", False)
                )
                stage_history.append(
                    {
                        "stage": f"irls_{irls_index}",
                        "relation_weight": float(current_relation_weight),
                        "movement_weight": float(current_movement_weight),
                        "nfev": int(getattr(solve_result, "nfev", -1)),
                        "cost": float(getattr(solve_result, "cost", np.nan)),
                        "termination": str(getattr(solve_result, "message", "")),
                        "verified": bool(verified),
                        "failed_relations": int(failed_count),
                        "jac_sparsity_used": bool(stage_jac_used),
                        "residual_size": int(stage_probe.size),
                        "inputs_beyond_tolerance": len(self._inputs_beyond_tolerance(completed_values)),
                        "elapsed_s": float(time.perf_counter() - t_stage),
                    }
                )
                # Stop once a pass stops reducing the number beyond tolerance:
                # further reweighting only churns (common for inconsistent data).
                new_beyond = len(self._inputs_beyond_tolerance(completed_values))
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

    solver = self._solver_report(
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
    validation = self._result_from_certificate(
        mode,
        certificate,
        termination=str(solve_result.message),
        solver=solver,
        include_values=True,
        extra={"uninitialized_free_variables": list(getattr(self, "_uninitialized_free_variables", []))},
    )

    # There is no separate candidate/final variable state. The latest solve output
    # becomes the current public value and is overwritten on every reconcile call.
    self._store_solved_values(completed_values)
    self._refresh_scales()
    stored_validation = verify_mode.run(self)
    if bool(stored_validation.get("success", False)) != bool(verified):
        validation["warnings"].append("stored values verify differently after public conversion")
    validation["variables"] = self.variables_by_name
    validation["likely_culprits"] = self._rank_input_culprits(validation.get("relation_status", {}))
    # The objective minimises this set: non-fixed inputs whose solved value left
    # their tolerance band, worst deviation first.
    validation["inputs_beyond_tolerance"] = self._inputs_beyond_tolerance(completed_values)
    return validation
