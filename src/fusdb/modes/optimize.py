"""Optimize mode."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from . import verify as verify_mode
from .verify import verify_values


def run(
    system: Any,
    *,
    objective: str | Callable[[dict[str, Any]], Any] | None = None,
    sense: str = "minimize",
    maxiter: int | None = None,
    movement_weight: float = 0.0,
    **_unused: Any,
) -> dict[str, Any]:
    """Run constrained optimization."""
    self = system
    result = self._new_result("optimize")
    if self._reject_unknown_options(result, _unused):
        return result
    if objective is None:
        result["errors"].append("optimize requires an objective variable or callable.")
        result["termination"] = "missing objective"
        return result
    try:
        x0, lower, upper, _x_scale, spans = self._pack_free_variables()
    except Exception as exc:
        result["errors"].append(str(exc))
        result["termination"] = "initialization failed"
        return result

    if self._record_uninitialized_failure(result):
        return result

    if x0.size == 0:
        return verify_mode.run(self)
    reference = self._values_from_variables(for_solver=True, skip_missing=True, use_input_values=True)

    def objective_value(x: np.ndarray) -> float:
        values = self._values_from_vector(x, spans)
        if callable(objective):
            raw = objective(values)
        else:
            raw = values[str(objective)]
        val = float(np.asarray(raw, dtype=float).reshape(-1)[0])
        if sense == "maximize":
            val = -val
        if movement_weight:
            move = self._movement_residuals(values, reference, spans)
            val += float(movement_weight) * float(np.dot(move, move))
        return val

    def equality_residual(x: np.ndarray) -> np.ndarray:
        values = self._values_from_vector(x, spans)
        _status, residuals, errors, _warnings = self._evaluate_relation_residuals(values, strict=True, solver_residuals=True)
        if errors:
            return np.full(max(1, residuals.size), 1.0e6, dtype=float)
        return residuals

    constraints = [NonlinearConstraint(equality_residual, -1e-8, 1e-8)]
    try:
        sol = minimize(
            objective_value,
            x0,
            method="trust-constr",
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"maxiter": maxiter or 200, "verbose": 0},
        )
    except Exception as exc:
        result["errors"].append(f"SciPy minimize failed: {exc}")
        result["termination"] = "solver error"
        return result
    values = self._values_from_vector(sol.x, spans)
    completed_values = self._complete_values(dict(values), strict=False)
    self._store_solved_values(completed_values)
    validation = verify_mode.run(self)
    validation.update({"mode": "optimize", "termination": str(sol.message), "solver": {"backend": "scipy.optimize.minimize", "success": bool(sol.success), "niter": int(getattr(sol, "nit", -1))}})
    return validation
