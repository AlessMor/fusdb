"""Species density and composition balance relations."""

import math
from typing import Any

import numpy as np

from fusdb import relation
from fusdb.registry import (
    SPECIES,
)
from ..utils import _positive_denominator

_IMPURITY_CHARGE = float(SPECIES["Imp"].atomic_number)

@relation(
    name="Ion density from tracked species densities",
    tags=("plasma", "composition"),
    outputs="n_i",
)
def ion_density_from_tracked_species_densities(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_imp: Any,
) -> Any:
    """Return total tracked ion density from species densities."""
    return n_D + n_T + n_He3 + n_He4 + n_imp


@relation(
    name="Electron density from tracked species densities",
    tags=("plasma", "composition"),
    outputs="n_e",
)
def electron_density_from_tracked_species_densities(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_imp: Any,
) -> Any:
    """Return electron density from charge neutrality."""
    return n_D + n_T + 2.0 * n_He3 + 2.0 * n_He4 + _IMPURITY_CHARGE * n_imp


@relation(
    name="D density from ion density and D fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_D",
)
def deuterium_density_from_ion_density_and_fraction(n_i: Any, f_D: Any) -> Any:
    """Return deuterium density from total ion density and D fraction."""
    return n_i * f_D


@relation(
    name="T density from ion density and T fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_T",
)
def tritium_density_from_ion_density_and_fraction(n_i: Any, f_T: Any) -> Any:
    """Return tritium density from total ion density and T fraction."""
    return n_i * f_T


@relation(
    name="He3 density from ion density and He3 fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_He3",
)
def helium3_density_from_ion_density_and_fraction(n_i: Any, f_He3: Any) -> Any:
    """Return helium-3 density from total ion density and He3 fraction."""
    return n_i * f_He3


@relation(
    name="He4 density from ion density and He4 fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_He4",
)
def helium4_density_from_ion_density_and_fraction(n_i: Any, f_He4: Any) -> Any:
    """Return helium-4 density from total ion density and He4 fraction."""
    return n_i * f_He4


def plasma_balance_ode(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
    *,
    injection_fractions: np.ndarray | tuple[float, float, float, float] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Return D/T/He3/He4 balances with implicit total-density fueling."""
    inv_tau_D = 0.0 if tau_p_D is None else 1.0 / tau_p_D
    inv_tau_T = 0.0 if tau_p_T is None else 1.0 / tau_p_T
    inv_tau_He3 = 0.0 if tau_p_He3 is None else 1.0 / tau_p_He3
    inv_tau_He4 = 0.0 if tau_p_He4 is None else 1.0 / tau_p_He4

    dn_D_dt = (
        -n_D * n_T * sigmav_DT
        - n_D**2 * (sigmav_DDn + sigmav_DDp)
        - n_D * n_He3 * sigmav_DHe3
        + n_T * n_He3 * sigmav_THe3_D
        - inv_tau_D * n_D
    )
    dn_T_dt = (
        +0.5 * n_D**2 * sigmav_DDp
        - n_D * n_T * sigmav_DT
        - n_T**2 * sigmav_TT
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - inv_tau_T * n_T
    )
    dn_He3_dt = (
        +0.5 * n_D**2 * sigmav_DDn
        - n_D * n_He3 * sigmav_DHe3
        - n_He3**2 * sigmav_He3He3
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - inv_tau_He3 * n_He3
    )
    dn_He4_dt = (
        +n_D * n_T * sigmav_DT
        + n_D * n_He3 * sigmav_DHe3
        + 0.5 * n_T**2 * sigmav_TT
        + 0.5 * n_He3**2 * sigmav_He3He3
        + n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - inv_tau_He4 * n_He4
    )

    total_density = n_D + n_T + n_He3 + n_He4
    total_density_safe = np.maximum(total_density, 1e-300)
    if injection_fractions is None:
        feed = np.stack([n_D, n_T, n_He3, n_He4], axis=0) / total_density_safe
    else:
        feed = np.asarray(injection_fractions, dtype=float)
        if feed.shape[0] != 4 or not np.isfinite(feed).all() or np.any(feed < 0.0):
            raise ValueError("injection_fractions must be a length-4 non-negative vector")
        feed = feed / _positive_denominator(np.sum(feed, axis=0), name="injection fraction sum")

    net_balance = dn_D_dt + dn_T_dt + dn_He3_dt + dn_He4_dt
    source = -net_balance * feed
    return dn_D_dt + source[0], dn_T_dt + source[1], dn_He3_dt + source[2], dn_He4_dt + source[3]


def _normalized_balances(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return normalized particle balances for residual relations."""
    balances = plasma_balance_ode(
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p_D,
        tau_p_T,
        tau_p_He3,
        tau_p_He4,
    )
    total_density = np.maximum(n_D + n_T + n_He3 + n_He4, 1e-300)
    return tuple(balance / total_density for balance in balances)


@relation(name="Steady-state D particle balance", tags=("plasma", "composition", "steady_state"))
def steady_state_deuterium_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
) -> Any:
    """Return normalized D particle-balance residual."""
    return _normalized_balances(
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p_D,
        tau_p_T,
        tau_p_He3,
        tau_p_He4,
    )[0]


@relation(name="Steady-state T particle balance", tags=("plasma", "composition", "steady_state"))
def steady_state_tritium_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
) -> Any:
    """Return normalized T particle-balance residual."""
    return _normalized_balances(
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p_D,
        tau_p_T,
        tau_p_He3,
        tau_p_He4,
    )[1]


@relation(name="Steady-state He3 particle balance", tags=("plasma", "composition", "steady_state"))
def steady_state_helium3_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
) -> Any:
    """Return normalized He3 particle-balance residual."""
    return _normalized_balances(
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p_D,
        tau_p_T,
        tau_p_He3,
        tau_p_He4,
    )[2]


@relation(name="Steady-state He4 particle balance", tags=("plasma", "composition", "steady_state"))
def steady_state_helium4_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
) -> Any:
    """Return normalized He4 particle-balance residual."""
    return _normalized_balances(
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p_D,
        tau_p_T,
        tau_p_He3,
        tau_p_He4,
    )[3]


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve steady-state composition for ordered/manual numerical evaluation."""
    from scipy.optimize import root

    profiles = (
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
    )
    arrays = [np.asarray(v, dtype=float) for v in profiles]
    if any(arr.ndim != 1 for arr in arrays):
        raise TypeError("Density and reactivity inputs must be 1D arrays.")
    if len({arr.size for arr in arrays}) != 1:
        raise ValueError("Density and reactivity profiles must all have the same length.")
    for name, tau in (
        ("tau_p_D", tau_p_D),
        ("tau_p_T", tau_p_T),
        ("tau_p_He3", tau_p_He3),
        ("tau_p_He4", tau_p_He4),
    ):
        if tau is not None and (float(tau) <= 0.0 or not math.isfinite(float(tau))):
            raise ValueError(f"{name} must be positive or None")

    n_points = arrays[0].size
    out = [np.zeros(n_points, dtype=float) for _ in range(4)]
    for i in range(n_points):
        seeded = np.asarray([arrays[0][i], arrays[1][i], arrays[2][i], arrays[3][i]], dtype=float)
        if not np.isfinite(seeded).all() or np.any(seeded < 0.0):
            raise ValueError("Seeded densities must be finite and non-negative.")
        total_density = float(np.sum(seeded))
        if total_density <= 0.0:
            continue
        initial_fractions = seeded / total_density

        def residual(fractions: np.ndarray) -> np.ndarray:
            fractions = np.asarray(fractions, dtype=float)
            state = total_density * fractions
            balances = plasma_balance_ode(
                state[0],
                state[1],
                state[2],
                state[3],
                arrays[4][i],
                arrays[5][i],
                arrays[6][i],
                arrays[7][i],
                arrays[8][i],
                arrays[9][i],
                arrays[10][i],
                arrays[11][i],
                tau_p_D,
                tau_p_T,
                tau_p_He3,
                tau_p_He4,
                injection_fractions=initial_fractions,
            )
            return np.asarray(
                [
                    balances[0] / total_density,
                    balances[1] / total_density,
                    balances[2] / total_density,
                    np.sum(fractions) - 1.0,
                ],
                dtype=float,
            )

        result = root(residual, initial_fractions, method=method, tol=tol, options={"maxfev": max_iter})
        solved_fraction = np.asarray(result.x, dtype=float)
        if not np.isfinite(solved_fraction).all() or np.any(solved_fraction < 0.0):
            raise RuntimeError("Steady-state composition solve produced an invalid state.")
        residual_vector = residual(solved_fraction)
        if float(np.linalg.norm(residual_vector)) > tol * 10.0:
            raise RuntimeError(f"Steady-state composition solve failed: {result.message!r}")
        solved_state = total_density * solved_fraction
        for j in range(4):
            out[j][i] = solved_state[j]
    return out[0], out[1], out[2], out[3]
