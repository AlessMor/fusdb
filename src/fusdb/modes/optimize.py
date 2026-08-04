"""Optimize mode."""

from __future__ import annotations

from collections.abc import Sequence
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
    constraints: Sequence[tuple[str, str, float | str]] | None = None,
    initial_guesses: Any = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Run constrained optimization.

    ``constraints`` adds per-run inequality bounds on any variable (a solver
    unknown or a completion-derived quantity such as ``greenwald_margin``),
    as ``(name, op, bound)`` triples with ``op`` in ``>= > <= < ==``.  ``bound``
    is a number for an absolute limit -- ``("P_fus", ">=", 2e9)`` -- or another
    variable name for a comparison -- ``("P_sep", ">=", "P_LH")`` (H-mode).
    They shape *this optimization only* and never touch the shared relation
    set, so a diagnostic margin can be turned into a limit for one study
    without enforcing it everywhere.  A lower bound on the objective (e.g.
    P_fus) also makes a degenerate low-power root infeasible, steering the
    optimizer to a physical operating point.
    """
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
    # A physical warm start matters here: the 0-D operating-point manifold is
    # multi-valued, so without a seed the optimizer can settle on a degenerate
    # (extinguished/runaway) root.  Merge caller guesses on top of the compiled
    # seeds, exactly as reconcile does, before packing.
    if initial_guesses:
        self.initial_guesses.update(
            {name: value for name, value in dict(initial_guesses).items() if name not in self.fixed and value is not None}
        )
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

    constraint_list = [NonlinearConstraint(equality_residual, -1e-8, 1e-8)]
    if constraints:
        # bound is a number (absolute) or a variable name (comparison); a
        # comparison constrains ``value(name) - value(bound)`` against 0.
        spec = [(str(name), str(op), bound, isinstance(bound, str)) for name, op, bound in constraints]
        lows, highs = [], []
        for _name, op, bound, is_var in spec:
            thresh = 0.0 if is_var else float(bound)
            if op in (">=", ">"):
                lows.append(thresh); highs.append(np.inf)
            elif op in ("<=", "<"):
                lows.append(-np.inf); highs.append(thresh)
            elif op == "==":
                lows.append(thresh - 1e-8); highs.append(thresh + 1e-8)
            else:
                result["errors"].append(f"optimize constraint op must be one of >= > <= < ==, got {op!r}.")
                result["termination"] = "invalid options"
                return result

        def constraint_values(x: np.ndarray) -> np.ndarray:
            # Complete so derived quantities (margins, etc.) resolve; a value
            # that cannot be produced yields NaN and trivially fails feasibility.
            values = self.complete(dict(self.unpack(x)))
            def val(n: str) -> float:
                return float(np.asarray(values.get(n, np.nan), dtype=float).reshape(-1)[0])
            return np.array(
                [val(name) - val(bound) if is_var else val(name) for name, _op, bound, is_var in spec],
                dtype=float,
            )

        constraint_list.append(NonlinearConstraint(constraint_values, np.array(lows), np.array(highs)))
    constraints = constraint_list
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
