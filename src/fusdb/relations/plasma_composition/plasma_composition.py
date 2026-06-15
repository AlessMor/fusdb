"""Density-based plasma composition relations.

This module is intended as a drop-in replacement for the current plasma
composition relation file.  It keeps the existing forward relations and adds
explicit inverse relations so a reconcile solve can grow from common reactor
inputs such as ``n_i``/``n_e`` plus ``f_D``/``f_T`` into species density profiles
needed by fusion-power relations.

Design goals:
    - Keep relations simple and algebraic.
    - Avoid nested solves inside relation functions.
    - Support scalars and 1-D profiles through NumPy broadcasting.
    - Make DT-only cases reachable from ``n_i`` or ``n_e`` and fuel fractions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from fusdb import relation
from fusdb.registry import SPECIES


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


_IMPURITY_CHARGE = float(SPECIES["Imp"].atomic_number)


def _positive_denominator(value: Any, *, name: str) -> Any:
    """Return a finite positive denominator for nonlinear least-squares.

    Relation functions should not abort during intermediate SciPy iterations.
    Domain/bounds and final residual checks decide whether the final state is
    acceptable.  This helper therefore clips invalid or non-positive temporary
    denominators to a tiny positive value instead of raising.
    """
    arr = np.asarray(value, dtype=float)
    arr = np.nan_to_num(arr, nan=1e-300, posinf=1e300, neginf=1e-300)
    arr = np.maximum(arr, 1e-300)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _finite_nonnegative(value: Any, *, name: str) -> Any:
    """Return a finite non-negative value for nonlinear least-squares."""
    arr = np.asarray(value, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e300, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _species_fraction(numerator: Any, denominator: Any, *, name: str) -> Any:
    """Return the integrated species fraction from density profiles.

    The ratio uses grid-integrated densities, so a profile whose edge value is
    exactly zero does not create an indeterminate pointwise ``0/0`` sample.
    For shape-proportional profiles this equals the pointwise fraction.
    """
    num = np.asarray(numerator, dtype=float).reshape(-1)
    den = np.asarray(denominator, dtype=float).reshape(-1)
    species = float(trapezoid(num)) if num.size > 1 else float(num[0])
    total = float(trapezoid(den)) if den.size > 1 else float(den[0])
    total = float(_positive_denominator(total, name=f"{name} denominator"))
    return species / total


def _zbar_from_fractions(
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return mean ion charge implied by ion fractions.

    The He/impurity fractions default to zero so common DT-only scenarios need
    only ``f_D`` and ``f_T``.  If those variables exist in the namespace,
    ``Relation.evaluate`` passes them despite the Python defaults.
    """
    return f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp


# ---------------------------------------------------------------------------
# Forward composition relations: species densities -> totals/fractions
# ---------------------------------------------------------------------------


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
    name="Integrated D fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_D",
)
def integrated_deuterium_fraction_from_density_profiles(n_D: Any, n_i: Any) -> Any:
    """Return pointwise deuterium fraction from density profiles."""
    return _species_fraction(n_D, n_i, name="f_D")


@relation(
    name="Integrated T fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_T",
)
def integrated_tritium_fraction_from_density_profiles(n_T: Any, n_i: Any) -> Any:
    """Return pointwise tritium fraction from density profiles."""
    return _species_fraction(n_T, n_i, name="f_T")


@relation(
    name="Integrated He3 fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_He3",
)
def integrated_helium3_fraction_from_density_profiles(n_He3: Any, n_i: Any) -> Any:
    """Return pointwise helium-3 fraction from density profiles."""
    return _species_fraction(n_He3, n_i, name="f_He3")


@relation(
    name="Integrated He4 fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_He4",
)
def integrated_helium4_fraction_from_density_profiles(n_He4: Any, n_i: Any) -> Any:
    """Return pointwise helium-4 fraction from density profiles."""
    return _species_fraction(n_He4, n_i, name="f_He4")


@relation(
    name="Integrated Imp fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_Imp",
)
def integrated_impurity_fraction_from_density_profiles(n_imp: Any, n_i: Any) -> Any:
    """Return pointwise impurity fraction from density profiles."""
    return _species_fraction(n_imp, n_i, name="f_Imp")


# ---------------------------------------------------------------------------
# Inverse composition relations: totals/fractions -> species densities
# ---------------------------------------------------------------------------


@relation(
    name="D density from ion density and D fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_D",
)
def deuterium_density_from_ion_density_and_fraction(n_i: Any, f_D: Any) -> Any:
    """Return deuterium density from total ion density and D fraction."""
    _finite_nonnegative(f_D, name="f_D")
    return n_i * f_D


@relation(
    name="T density from ion density and T fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_T",
)
def tritium_density_from_ion_density_and_fraction(n_i: Any, f_T: Any) -> Any:
    """Return tritium density from total ion density and T fraction."""
    _finite_nonnegative(f_T, name="f_T")
    return n_i * f_T


@relation(
    name="He3 density from ion density and He3 fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_He3",
)
def helium3_density_from_ion_density_and_fraction(n_i: Any, f_He3: Any) -> Any:
    """Return helium-3 density from total ion density and He3 fraction."""
    _finite_nonnegative(f_He3, name="f_He3")
    return n_i * f_He3


@relation(
    name="He4 density from ion density and He4 fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_He4",
)
def helium4_density_from_ion_density_and_fraction(n_i: Any, f_He4: Any) -> Any:
    """Return helium-4 density from total ion density and He4 fraction."""
    _finite_nonnegative(f_He4, name="f_He4")
    return n_i * f_He4


@relation(
    name="Impurity density from ion density and impurity fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_imp",
)
def impurity_density_from_ion_density_and_fraction(n_i: Any, f_Imp: Any) -> Any:
    """Return impurity density from total ion density and impurity fraction."""
    _finite_nonnegative(f_Imp, name="f_Imp")
    return n_i * f_Imp


@relation(
    name="Ion density from electron density and fuel fractions",
    tags=("plasma", "composition", "inverse"),
    outputs="n_i",
)
def ion_density_from_electron_density_and_fuel_fractions(
    n_e: Any,
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return total ion density from electron density and ion fractions.

    This makes DT-only cases reachable from ``n_e``, ``f_D`` and ``f_T``.
    Additional He/impurity fractions are used when present in the namespace.
    """
    zbar = _positive_denominator(
        _zbar_from_fractions(f_D, f_T, f_He3, f_He4, f_Imp),
        name="mean ion charge",
    )
    return n_e / zbar


@relation(
    name="Electron density from ion density and fuel fractions",
    tags=("plasma", "composition"),
    outputs="n_e",
)
def electron_density_from_ion_density_and_fuel_fractions(
    n_i: Any,
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return electron density from ion density and ion fractions.

    This gives a direct consistency check for cases that supply both ``n_i``
    and ``n_e`` without requiring all individual minority densities.
    """
    zbar = _positive_denominator(
        _zbar_from_fractions(f_D, f_T, f_He3, f_He4, f_Imp),
        name="mean ion charge",
    )
    return n_i * zbar


@relation(
    name="Average fuel mass number",
    tags=("plasma", "composition"),
    outputs="afuel",
)
def average_fuel_mass_number(f_D: Any, f_T: Any, f_He3: Any = 0.0) -> Any:
    """Return average fuel mass number from fuel fractions.

    ``f_He3`` defaults to zero so DT cases can compute ``afuel`` from only
    ``f_D`` and ``f_T``.  If ``f_He3`` exists in the solve namespace, it is used.
    """
    fuel_total = _positive_denominator(f_D + f_T + f_He3, name="fuel ion inventory")
    numerator = (
        f_D * float(SPECIES["D"].atomic_mass)
        + f_T * float(SPECIES["T"].atomic_mass)
        + f_He3 * float(SPECIES["He3"].atomic_mass)
    )
    return numerator / fuel_total


# ---------------------------------------------------------------------------
# Steady-state particle-balance residuals
# ---------------------------------------------------------------------------


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


def _normalized_impurity_balance(n_imp: Any, tau_p_Imp: Any, n_i: Any) -> Any:
    """Return the normalized impurity balance residual."""
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


@relation(name="Steady-state Imp particle balance", tags=("plasma", "composition", "steady_state"))
def steady_state_impurity_balance(n_imp: Any, tau_p_Imp: Any, n_i: Any) -> Any:
    """Return normalized impurity particle-balance residual."""
    return _normalized_impurity_balance(n_imp, tau_p_Imp, n_i)


# ---------------------------------------------------------------------------
# Manual helper for ordered/non-solver workflows.  Not decorated.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# cfspopcon ports (UNDECORATED scaffolds) — source:
# cfspopcon/formulas/impurities/zeff_and_dilution_from_impurities.py and
# cfspopcon/formulas/plasma_pressure/plasma_temperature.py
# These compute impurity dilution / Z_eff and the average ion temperature, which
# fusdb does not yet have. Review formula + variable/unit mapping (average_electron_*
# -> n_avg/T_avg, average_ion_density -> n_i_avg, z_effective -> Z_eff). The dilution
# function needs per-species atomic-data charge states (calc_impurity_charge_state,
# Radas) and xarray species arrays; adapt to the fusdb single-"Imp" model before
# decorating with @relation to activate.
# ---------------------------------------------------------------------------


# TODO(cfspopcon): activate as a fusdb relation (average_ion_temp output).
def calc_average_ion_temp_from_temperature_ratio(average_electron_temp, ion_to_electron_temp_ratio):
    """cfspopcon: average_ion_temp = average_electron_temp * ion_to_electron_temp_ratio."""
    return average_electron_temp * ion_to_electron_temp_ratio


# TODO(cfspopcon): helper for calc_zeff_and_dilution_due_to_impurities.
def calc_change_in_zeff(impurity_charge_state, impurity_concentration):
    """cfspopcon: change in Z_eff = Z*(Z-1)*c_imp."""
    return impurity_charge_state * (impurity_charge_state - 1.0) * impurity_concentration


# TODO(cfspopcon): helper for calc_zeff_and_dilution_due_to_impurities.
def calc_change_in_dilution(impurity_charge_state, impurity_concentration):
    """cfspopcon: change in n_fuel/n_e = Z*c_imp."""
    return impurity_charge_state * impurity_concentration


# TODO(cfspopcon): activate as fusdb relation(s) (z_effective, dilution, average_ion_density,
#   ...). Needs calc_impurity_charge_state (atomic-data/Radas) and xarray species arrays.
def calc_zeff_and_dilution_due_to_impurities(
    average_electron_density,
    average_electron_temp,
    impurity_concentration,
    atomic_data,
):
    """cfspopcon: impact of core impurities on Z_eff and dilution.

    Returns (impurity_charge_state, change_in_zeff, change_in_dilution, z_effective,
    dilution, summed_impurity_density, average_ion_density).
    """
    from cfspopcon.formulas.impurities.impurity_charge_state import (  # noqa: F401  # TODO: atomic data
        calc_impurity_charge_state,
    )

    starting_zeff = 1.0
    starting_dilution = 1.0

    impurity_charge_state = calc_impurity_charge_state(
        average_electron_density, average_electron_temp, impurity_concentration, atomic_data
    )
    change_in_zeff = calc_change_in_zeff(impurity_charge_state, impurity_concentration)
    change_in_dilution = calc_change_in_dilution(impurity_charge_state, impurity_concentration)

    z_effective = starting_zeff + change_in_zeff.sum(dim="dim_species")
    dilution = starting_dilution - change_in_dilution.sum(dim="dim_species")
    dilution = dilution.where(dilution >= 0, 0.0)
    summed_impurity_density = impurity_concentration.sum(dim="dim_species") * average_electron_density
    average_ion_density = dilution * average_electron_density

    return (
        impurity_charge_state,
        change_in_zeff,
        change_in_dilution,
        z_effective,
        dilution,
        summed_impurity_density,
        average_ion_density,
    )
