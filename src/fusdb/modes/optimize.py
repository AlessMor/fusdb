"""Optimize mode."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from . import verify as verify_mode
from ._common import new_result, record_uninitialized_failure, reject_unknown_options


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
    result = new_result(self, "optimize")
    if reject_unknown_options(result, _unused):
        return result
    if sense not in {"minimize", "maximize"}:
        result["errors"].append(f"optimize sense must be 'minimize' or 'maximize', got {sense!r}.")
        result["termination"] = "invalid options"
        return result
    if objective is None:
        result["errors"].append("optimize requires an objective variable or callable.")
        result["termination"] = "missing objective"
        return result
    try:
        x0, lower, upper = self.pack()
    except Exception as exc:
        result["errors"].append(str(exc))
        result["termination"] = "initialization failed"
        return result

    if record_uninitialized_failure(self, result):
        return result

    if x0.size == 0:
        validation = verify_mode.run(self)
        validation["mode"] = "optimize"
        validation["termination"] = "no free variables; validation only"
        return validation

    # One frozen residual-row layout for the whole solve (the single residual
    # protocol, see RelationSystem.residual_layout): every constraint and
    # movement evaluation fills exactly these rows, so a value that goes
    # missing penalizes its own rows instead of changing the vector size.
    try:
        layout = self.residual_layout(self.unpack(x0), include_movement=bool(movement_weight))
    except Exception as exc:
        result["errors"].append(f"Residual initialization failed: {exc}")
        result["termination"] = "initialization failed"
        return result

    def objective_value(x: np.ndarray) -> float:
        values = self.unpack(x)
        if callable(objective):
            raw = objective(values)
        else:
            raw = values[str(objective)]
        val = float(np.asarray(raw, dtype=float).reshape(-1)[0])
        if sense == "maximize":
            val = -val
        if movement_weight:
            move = self.layout_movement_rows(values, layout)
            val += float(movement_weight) * float(np.dot(move, move))
        return val

    def equality_residual(x: np.ndarray) -> np.ndarray:
        values = self.unpack(x)
        return self.layout_relation_rows(values, layout)

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
    values = self.unpack(sol.x)
    completed_values = self.complete(dict(values))
    self.store(completed_values)
    validation = verify_mode.run(self)
    validation.update({"mode": "optimize", "termination": str(sol.message), "solver": {"backend": "scipy.optimize.minimize", "success": bool(sol.success), "niter": int(getattr(sol, "nit", -1))}})
    return validation
