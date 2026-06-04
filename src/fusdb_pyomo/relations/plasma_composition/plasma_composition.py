"""Density-based plasma composition relations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fusdb_pyomo import relation
from fusdb_pyomo.registry import SPECIES


def _positive_denominator(value: Any, *, name: str) -> Any:
    """Return a denominator after checking numeric positivity.

    Args:
        value: Candidate denominator.
        name: Name used in error messages.

    Returns:
        Original denominator value.
    """
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _species_fraction(numerator: Any, denominator: Any, *, name: str) -> Any:
    """Return a core species fraction from tracked ion densities.

    Args:
        numerator: Species density profile or scalar.
        denominator: Total tracked ion density profile or scalar.
        name: Fraction name used in validation messages.

    Returns:
        Pointwise species fraction.
    """
    # Reject zero or negative ion density before forming the composition ratio.
    denominator = _positive_denominator(denominator, name=f"{name} denominator")
    return numerator / denominator


@relation(
    name="Ion density from tracked species densities",
    tags=("plasma",),
    outputs="n_i",
)
def ion_density_from_tracked_species_densities(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    n_imp: float,
) -> Any:
    """Return the total tracked ion density profile.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        n_imp: Generic impurity density.

    Returns:
        Total tracked ion density.
    """
    return n_D + n_T + n_He3 + n_He4 + n_imp


@relation(
    name="Electron density from tracked species densities",
    tags=("plasma",),
    outputs="n_e",
)
def electron_density_from_tracked_species_densities(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    n_imp: float,
) -> Any:
    """Return the electron density profile from charge neutrality.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        n_imp: Generic impurity density.

    Returns:
        Electron density.
    """
    z_imp = float(SPECIES["Imp"].atomic_number)
    return n_D + n_T + 2.0 * n_He3 + 2.0 * n_He4 + z_imp * n_imp


@relation(
    name="Integrated D fraction from density profiles",
    tags=("plasma",),
    outputs="f_D",
)
def integrated_deuterium_fraction_from_density_profiles(n_D: float, n_i: float) -> Any:
    """Return the core deuterium fraction from tracked ion density.

    Args:
        n_D: Deuterium density.
        n_i: Total tracked ion density.
    Returns:
        Pointwise deuterium fraction.
    """
    return _species_fraction(n_D, n_i, name="f_D")


@relation(
    name="Integrated T fraction from density profiles",
    tags=("plasma",),
    outputs="f_T",
)
def integrated_tritium_fraction_from_density_profiles(n_T: float, n_i: float) -> Any:
    """Return the core tritium fraction from tracked ion density.

    Args:
        n_T: Tritium density.
        n_i: Total tracked ion density.
    Returns:
        Pointwise tritium fraction.
    """
    return _species_fraction(n_T, n_i, name="f_T")


@relation(
    name="Integrated He3 fraction from density profiles",
    tags=("plasma",),
    outputs="f_He3",
)
def integrated_helium3_fraction_from_density_profiles(n_He3: float, n_i: float) -> Any:
    """Return the core helium-3 fraction from tracked ion density.

    Args:
        n_He3: Helium-3 density.
        n_i: Total tracked ion density.
    Returns:
        Pointwise helium-3 fraction.
    """
    return _species_fraction(n_He3, n_i, name="f_He3")


@relation(
    name="Integrated He4 fraction from density profiles",
    tags=("plasma",),
    outputs="f_He4",
)
def integrated_helium4_fraction_from_density_profiles(n_He4: float, n_i: float) -> Any:
    """Return the core helium-4 fraction from tracked ion density.

    Args:
        n_He4: Helium-4 density.
        n_i: Total tracked ion density.
    Returns:
        Pointwise helium-4 fraction.
    """
    return _species_fraction(n_He4, n_i, name="f_He4")


@relation(
    name="Integrated Imp fraction from density profiles",
    tags=("plasma",),
    outputs="f_Imp",
)
def integrated_impurity_fraction_from_density_profiles(n_imp: float, n_i: float) -> Any:
    """Return the core impurity fraction from tracked ion density.

    Args:
        n_imp: Generic impurity density.
        n_i: Total tracked ion density.
    Returns:
        Pointwise impurity fraction.
    """
    return _species_fraction(n_imp, n_i, name="f_Imp")


@relation(
    name="Average fuel mass number",
    tags=("plasma",),
    outputs="afuel",
)
def average_fuel_mass_number(f_D: float, f_T: float, f_He3: float) -> Any:
    """Return the average fuel mass number from core fuel fractions.

    Args:
        f_D: Deuterium fraction.
        f_T: Tritium fraction.
        f_He3: Helium-3 fraction.

    Returns:
        Fuel-only average ion mass number.
    """
    fuel_total = _positive_denominator(f_D + f_T + f_He3, name="fuel ion inventory")
    numerator = (
        f_D * float(SPECIES["D"].atomic_mass)
        + f_T * float(SPECIES["T"].atomic_mass)
        + f_He3 * float(SPECIES["He3"].atomic_mass)
    )
    return numerator / fuel_total


def plasma_balance_ode(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    sigmav_DT: float,
    sigmav_DDn: float,
    sigmav_DDp: float,
    sigmav_DHe3: float,
    sigmav_TT: float,
    sigmav_He3He3: float,
    sigmav_THe3_D: float,
    sigmav_THe3_np: float,
    tau_p_D: float | None,
    tau_p_T: float | None,
    tau_p_He3: float | None,
    tau_p_He4: float | None,
    *,
    injection_fractions: np.ndarray | tuple[float, float, float, float] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Return D/T/He3/He4 balances with implicit total-density fueling.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        sigmav_DT: DT reactivity.
        sigmav_DDn: DDn reactivity.
        sigmav_DDp: DDp reactivity.
        sigmav_DHe3: D-He3 reactivity.
        sigmav_TT: TT reactivity.
        sigmav_He3He3: He3-He3 reactivity.
        sigmav_THe3_D: T-He3 alpha+D branch reactivity.
        sigmav_THe3_np: T-He3 alpha+n+p branch reactivity.
        tau_p_D: Deuterium particle confinement time.
        tau_p_T: Tritium particle confinement time.
        tau_p_He3: Helium-3 particle confinement time.
        tau_p_He4: Helium-4 particle confinement time.
        injection_fractions: Optional fixed D/T/He3/He4 fueling split.

    Returns:
        Four particle balance values with total-density source included.
    """
    inv_tau_D = 0.0 if tau_p_D is None else 1.0 / tau_p_D
    inv_tau_T = 0.0 if tau_p_T is None else 1.0 / tau_p_T
    inv_tau_He3 = 0.0 if tau_p_He3 is None else 1.0 / tau_p_He3
    inv_tau_He4 = 0.0 if tau_p_He4 is None else 1.0 / tau_p_He4

    dn_D_dt = (
        - n_D * n_T * sigmav_DT
        - n_D**2 * (sigmav_DDn + sigmav_DDp)
        - n_D * n_He3 * sigmav_DHe3
        + n_T * n_He3 * sigmav_THe3_D
        - inv_tau_D * n_D
    )
    dn_T_dt = (
        + 0.5 * n_D**2 * sigmav_DDp
        - n_D * n_T * sigmav_DT
        - n_T**2 * sigmav_TT
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - inv_tau_T * n_T
    )
    dn_He3_dt = (
        + 0.5 * n_D**2 * sigmav_DDn
        - n_D * n_He3 * sigmav_DHe3
        - n_He3**2 * sigmav_He3He3
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - inv_tau_He3 * n_He3
    )
    dn_He4_dt = (
        + n_D * n_T * sigmav_DT
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


def _normalized_impurity_balance(n_imp: Any, tau_p_Imp: Any, n_i: Any) -> Any:
    """Return the normalized impurity balance residual.

    Args:
        n_imp: Generic impurity density.
        tau_p_Imp: Generic impurity confinement time.
        n_i: Total tracked ion density.

    Returns:
        Normalized impurity density time derivative.
    """
    # Model the impurity channel as confinement-loss only until a source model exists.
    return -(n_imp / tau_p_Imp) / np.maximum(n_i, 1e-300)


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
        n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3,
        sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np, tau_p_D, tau_p_T, tau_p_He3, tau_p_He4
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
        n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3,
        sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np, tau_p_D, tau_p_T, tau_p_He3, tau_p_He4
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
        n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3,
        sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np, tau_p_D, tau_p_T, tau_p_He3, tau_p_He4
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
        n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3,
        sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np, tau_p_D, tau_p_T, tau_p_He3, tau_p_He4
    )[3]


@relation(name="Steady-state Imp particle balance", tags=("plasma", "composition", "steady_state"))
def steady_state_impurity_balance(n_imp: Any, tau_p_Imp: Any, n_i: Any) -> Any:
    """Return normalized impurity particle-balance residual.

    Args:
        n_imp: Generic impurity density.
        tau_p_Imp: Generic impurity confinement time.
        n_i: Total tracked ion density.

    Returns:
        Normalized impurity particle-balance residual.
    """
    # Reuse the explicit impurity residual so the steady-state relation stays simple.
    return _normalized_impurity_balance(n_imp, tau_p_Imp, n_i)


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
    """Solve steady-state composition for ordered/manual numerical evaluation.

    This function is intentionally not decorated.  Simultaneous solve modes use
    the particle-balance residual relations above instead of a nested root solve.
    """
    from scipy.optimize import root

    profiles = (
        n_D, n_T, n_He3, n_He4, sigmav_DT, sigmav_DDn, sigmav_DDp, sigmav_DHe3,
        sigmav_TT, sigmav_He3He3, sigmav_THe3_D, sigmav_THe3_np,
    )
    arrays = [np.asarray(v, dtype=float) for v in profiles]
    if any(arr.ndim != 1 for arr in arrays):
        raise TypeError("Density and reactivity inputs must be 1D arrays.")
    if len({arr.size for arr in arrays}) != 1:
        raise ValueError("Density and reactivity profiles must all have the same length.")
    for name, tau in (("tau_p_D", tau_p_D), ("tau_p_T", tau_p_T), ("tau_p_He3", tau_p_He3), ("tau_p_He4", tau_p_He4)):
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
                state[0], state[1], state[2], state[3], arrays[4][i], arrays[5][i], arrays[6][i], arrays[7][i],
                arrays[8][i], arrays[9][i], arrays[10][i], arrays[11][i], tau_p_D, tau_p_T, tau_p_He3, tau_p_He4,
                injection_fractions=initial_fractions,
            )
            return np.asarray([balances[0] / total_density, balances[1] / total_density, balances[2] / total_density, np.sum(fractions) - 1.0], dtype=float)

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
