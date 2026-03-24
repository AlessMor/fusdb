"""Steady-state plasma composition relation on tracked ion densities."""

from __future__ import annotations

import math

import numpy as np

from fusdb.relation_util import relation


@relation(
    name="Steady-state plasma composition",
    inputs=(
        "n_D", "n_T","n_He3", "n_He4", 
        "sigmav_DT", "sigmav_DDn", "sigmav_DDp", "sigmav_DHe3", "sigmav_TT", "sigmav_He3He3", "sigmav_THe3_D", "sigmav_THe3_np",
        "tau_p_D", "tau_p_T", "tau_p_He3", "tau_p_He4",
    ),
    outputs=("n_D", "n_T", "n_He3", "n_He4"),
    tags=("plasma",),
    rel_tol_default=1e-10,
    abs_tol_default=1e-12,
)
def steady_state_plasma_composition(
    n_D: np.ndarray,
    n_T: np.ndarray,
    n_He3: np.ndarray,
    n_He4: np.ndarray,
    sigmav_DT: np.ndarray,
    sigmav_DDn: np.ndarray,
    sigmav_DDp: np.ndarray,
    sigmav_DHe3: np.ndarray,
    sigmav_TT: np.ndarray,
    sigmav_He3He3: np.ndarray,
    sigmav_THe3_D: np.ndarray,
    sigmav_THe3_np: np.ndarray,
    tau_p_D: float | None,
    tau_p_T: float | None,
    tau_p_He3: float | None,
    tau_p_He4: float | None,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    method: str = "hybr",
) -> dict[str, np.ndarray]:
    """
    Solve the steady-state D/T/He3/He4 density bundle on one radial grid.
    NOTE: add n_X, sigmav_R if additional species are considered in the future
    """
    from scipy.optimize import root

    # check species and reactivity inputs for consistency before solving
    if not all(isinstance(value, np.ndarray) and value.ndim == 1 for value in (n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3, sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np,)):
        raise TypeError("Density and reactivity inputs must already be 1D numpy arrays before relation evaluation.")
    if len({value.size for value in (n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3, sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np,)}) != 1:
        raise ValueError(f"Density and reactivity profiles must all have the same length")

    n_points = n_D.size
    tau_D, tau_T, tau_He3, tau_He4 = (
        math.inf if tau is None else float(tau)
        for tau in (tau_p_D, tau_p_T, tau_p_He3, tau_p_He4)
    )
    out_D = np.zeros_like(n_D, dtype=float)
    out_T = np.zeros_like(n_T, dtype=float)
    out_He3 = np.zeros_like(n_He3, dtype=float)
    out_He4 = np.zeros_like(n_He4, dtype=float)

    # Solve one independent nonlinear steady-state system at each radial point.
    for i in range(n_points):
        seed = np.asarray([n_D[i], n_T[i], n_He3[i], n_He4[i]], dtype=float)
        total_n = float(np.sum(seed))
        if total_n <= 0.0:
            continue

        density_scale = max(total_n, 1.0)

        def _rhs(n: np.ndarray) -> np.ndarray:
            n_D, n_T, n_He3, n_He4 = (float(value) for value in n)
            rhs = np.asarray(
                [   
                    # n_D, n_T, n_He3, n_He4
                    -n_D * n_T * sigmav_DT[i] - n_D**2 * (sigmav_DDn[i] + sigmav_DDp[i]) - n_D * n_He3 * sigmav_DHe3[i] + n_T * n_He3 * sigmav_THe3_D[i] - (0.0 if not math.isfinite(tau_D) else n_D / tau_D),
                    +0.5 * n_D**2 * sigmav_DDp[i] - n_D * n_T * sigmav_DT[i] - n_T**2 * sigmav_TT[i] - n_T * n_He3 * (sigmav_THe3_D[i] + sigmav_THe3_np[i]) - (0.0 if not math.isfinite(tau_T) else n_T / tau_T),
                    +0.5 * n_D**2 * sigmav_DDn[i] - n_D * n_He3 * sigmav_DHe3[i] - n_He3**2 * sigmav_He3He3[i] - n_T * n_He3 * (sigmav_THe3_D[i] + sigmav_THe3_np[i]) - (0.0 if not math.isfinite(tau_He3) else n_He3 / tau_He3),
                    +n_D * n_T * sigmav_DT[i] + n_D * n_He3 * sigmav_DHe3[i] + 0.5 * n_T**2 * sigmav_TT[i] + 0.5 * n_He3**2 * sigmav_He3He3[i] + n_T * n_He3 * (sigmav_THe3_D[i] + sigmav_THe3_np[i]) - (0.0 if not math.isfinite(tau_He4) else n_He4 / tau_He4),
                ],
                dtype=float,
            )
            return rhs - float(np.sum(rhs)) * seed / total_n

        initial_state = np.clip(
            seed + _rhs(seed),
            0.0,
            np.inf,
        )
        initial_sum = float(np.sum(initial_state))
        if initial_sum > 0.0:
            initial_state *= total_n / initial_sum
        else:
            initial_state = seed

        def _residual(state: np.ndarray) -> np.ndarray:
            # Solve directly for the four densities under one explicit total-density constraint.
            state = np.asarray(state, dtype=float)
            if state.shape != (4,) or not np.isfinite(state).all() or np.any(state < 0.0):
                penalty = 1e3 + float(np.sum(np.abs(state)))
                return np.full(4, penalty, dtype=float)

            residual = _rhs(state)
            return np.asarray(
                [
                    residual[0] / density_scale,
                    residual[1] / density_scale,
                    residual[2] / density_scale,
                    (float(np.sum(state)) - total_n) / density_scale,
                ],
                dtype=float,
            )

        # Solve the direct density system without introducing fraction unknowns.
        result = root(
            _residual,
            initial_state,
            method=method,
            tol=tol,
            options={"maxfev": max_iter},
        )
        if not result.success:
            residual_norm = float(np.linalg.norm(_residual(result.x)))
            raise RuntimeError(
                f"Steady-state composition solve failed: {result.message} (||res||={residual_norm:.3e})"
            )
        solved_state = np.asarray(result.x, dtype=float)
        if solved_state.shape != (4,) or not np.isfinite(solved_state).all() or np.any(solved_state < 0.0):
            raise RuntimeError("Steady-state composition solve produced an invalid state")

        # Reject weakly converged solutions even if the nonlinear solver reported success.
        residual_norm = float(np.linalg.norm(_residual(solved_state)))
        density_error = abs(float(np.sum(solved_state)) - total_n)
        if not math.isfinite(residual_norm) or residual_norm > 1e-10 or density_error > max(1e-12 * total_n, 1e-12):
            raise RuntimeError(
                "Steady-state composition solve did not converge tightly enough "
                f"(||res||={residual_norm:.3e}, density_error={density_error:.3e})"
            )

        out_D[i], out_T[i], out_He3[i], out_He4[i] = solved_state

    return {"n_D": out_D, "n_T": out_T, "n_He3": out_He3, "n_He4": out_He4}
