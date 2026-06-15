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
    relation_weight: float = 1.0,
    relation_weight_schedule: Iterable[float] | None = None,
    verbose: int = 0,
    compute_missing: bool = True,
    initial_residual_tol: float = 1.0,
    **_unused: Any,
) -> dict[str, Any]:
    """Run structural simultaneous reconciliation."""
    self = system
    mode = "reconcile"
    result = self._new_result(mode)
    if self._reject_unknown_options(result, _unused):
        return result
    initial_values: dict[str, Any] = {}
    self._initial_guesses = {}
    if compute_missing:
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
        completed_values = current_certificate["values"]
        validation = self._new_result(mode)
        validation.update(
            {
                "relation_status": current_certificate["relation_status"],
                "residuals": current_certificate["residuals"].tolist(),
                "errors": current_certificate["errors"],
                "warnings": current_certificate["warnings"],
                "variable_status": self._classify_variables(current_certificate["relation_status"]),
                "termination": "already verified; no reconcile solve",
                "success": True,
                "verified": True,
                "certificate": {k: v for k, v in current_certificate.items() if k not in {"residuals", "values"}},
                "variables": self.variables_by_name,
                "relations": self.primary_relations,
                "graph": self.graph,
                "compiler_report": self.compiler_report,
                "values": completed_values,
                "uninitialized_free_variables": [],
                "solver": {
                    "backend": "none",
                    "success": True,
                    "status": 0,
                    "cost": 0.0,
                    "optimality": 0.0,
                    "nfev": 0,
                    "message": "already verified; no reconcile solve",
                    "residual_calls": 0,
                    "residual_eval_time_s": 0.0,
                    "residual_size": int(current_certificate["residuals"].size),
                    "solver_dim": 0,
                    "jac_sparsity_used": False,
                    "jac_sparsity_shape": None,
                    "residual_eval_mean_ms": 0.0,
                    "relation_weight": float(relation_weight),
                    "relation_weight_schedule": [],
                    "phase_schedule": [],
                    "stage_history": [],
                    "movement_weight": float(movement_weight),
                    "initial_guess_variables": int(len(initial_values)),
                },
            }
        )
        return validation

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
        # these systems need roughly O(dim) evaluations to converge, so the
        # budget scales with the packed dimension with a floor that lets small
        # well-posed systems finish.  Genuinely inconsistent cases still stop
        # at the gtol/ftol plateau well before exhausting this budget; callers
        # can pass max_nfev explicitly to tighten or extend the search.
        max_nfev = int(min(40, max(15, 3 * int(x0.size))))
    if x0.size == 0:
        validation = verify_mode.run(self)
        validation["mode"] = mode
        validation["termination"] = "no free variables; validation only"
        return validation

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
            values = self._values_from_vector(x, spans)
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
        solved = self._values_from_vector(x, spans)
        certificate = verify_values(self, solved, complete=True)
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
    residuals = np.empty(0, dtype=float)
    errors: list[str] = []
    warnings: list[str] = []
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
            # the completion DAG and is conservative (it never omits a real
            # dependency).  It is accepted only when its shape exactly matches
            # the live residual/variable sizes for this stage; any mismatch
            # falls back to dense differences, so a stale pattern can never make
            # the solve wrong -- only slower.
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
            relation_status, residuals, errors, warnings, verified, completed_values, certificate = verify_candidate(current_x)
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
    except Exception as exc:
        result["errors"].append(f"SciPy least_squares failed: {exc}")
        result["termination"] = "solver error"
        return result
    if solve_result is None:
        result["errors"].append("SciPy least_squares did not run.")
        result["termination"] = "solver error"
        return result

    validation = self._new_result(mode)
    validation.update(
        {
            "relation_status": relation_status,
            "residuals": residuals.tolist(),
            "errors": errors,
            "warnings": warnings,
            "variable_status": self._classify_variables(relation_status),
            "termination": str(solve_result.message),
            "success": bool(verified),
            "verified": bool(verified),
            "certificate": {k: v for k, v in certificate.items() if k not in {"residuals", "values"}},
            "variables": self.variables_by_name,
            "relations": self.primary_relations,
            "graph": self.graph,
            "compiler_report": self.compiler_report,
            "values": completed_values,
            "uninitialized_free_variables": list(getattr(self, "_uninitialized_free_variables", [])),
            "solver": {
                "backend": "scipy.optimize.least_squares",
                "success": bool(solve_result.success),
                "status": int(getattr(solve_result, "status", 0)),
                "cost": float(getattr(solve_result, "cost", np.nan)),
                "optimality": float(getattr(solve_result, "optimality", np.nan)),
                "nfev": int(getattr(solve_result, "nfev", -1)),
                "message": str(solve_result.message),
                "residual_calls": int(residual_calls),
                "residual_eval_time_s": float(residual_eval_time_s),
                "residual_size": int(final_probe_size),
                "solver_dim": int(x0.size),
                "jac_sparsity_used": bool(jac_sparsity_used),
                "jac_sparsity_shape": tuple(jac_sparsity.shape) if jac_sparsity is not None else None,
                "residual_eval_mean_ms": float(1000.0 * residual_eval_time_s / max(residual_calls, 1)),
                "relation_weight": float(current_relation_weight),
                "relation_weight_schedule": [float(item) for item in weight_schedule],
                "phase_schedule": [{"relation_weight": float(rw), "movement_weight": float(mw)} for rw, mw in phase_schedule],
                "stage_history": stage_history,
                "movement_weight": float(current_movement_weight),
                "initial_guess_variables": int(len(initial_values)),
            },
        }
    )

    # There is no separate candidate/final variable state. The latest solve output
    # becomes the current public value and is overwritten on every reconcile call.
    self._store_solved_values(completed_values)
    self._refresh_scales()
    stored_validation = verify_mode.run(self)
    if bool(stored_validation.get("success", False)) != bool(verified):
        validation["warnings"].append("stored values verify differently after public conversion")
    validation["variables"] = self.variables_by_name
    validation["likely_culprits"] = self._rank_input_culprits(validation.get("relation_status", {}), validation.get("variable_status", {}))
    return validation
