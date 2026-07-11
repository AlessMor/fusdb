"""Profile peaking and parabolic profile relations."""

from functools import lru_cache
from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import volume_average


# From plasma_profiles/density_peaking.py
# TODO(low): from cfspopcon add
    # density peaking for tokamaks Equation 3 from p1334 of Angioni et al.
    # different forms are used

# From plasma_profiles/temperature_peaking.py
# TODO(low): add from cfspopcon


@lru_cache(maxsize=16)
def _peaking_table(rho_key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return a monotone ``(peakings, alphas)`` table for one rho grid.

    ``peak/volume_average`` of ``(1-rho^2)^alpha`` is strictly increasing in
    alpha, so one precomputed table per grid inverts the peaking->alpha map in
    O(log n) by interpolation instead of an 80-step bisection on every profile
    build.
    """
    rho = np.asarray(rho_key, dtype=float)
    base = np.maximum(1.0 - rho**2, 0.0)
    alphas = np.linspace(0.0, 50.0, 2001)
    peaks = np.empty_like(alphas)
    for i, alpha in enumerate(alphas):
        shape = base**alpha
        mean = float(volume_average(shape, rho))
        peaks[i] = shape[0] / max(mean, 1e-300)
    return peaks, alphas


def _alpha_for_peaking(peaking: float, rho: np.ndarray) -> float:
    """Return alpha for shape=(1-rho^2)^alpha with requested peak/average."""
    target = max(float(peaking), 1.0)
    if target <= 1.0 + 1e-12:
        return 0.0
    peaks, alphas = _peaking_table(tuple(float(v) for v in np.asarray(rho, dtype=float)))
    if target >= peaks[-1]:
        return float(alphas[-1])
    return float(np.interp(target, peaks, alphas))


# Small pedestal so a generated profile never reaches exactly zero at rho=1.
# A real plasma has finite separatrix values; the exact zero of (1-rho^2)^alpha
# makes temperature-driven relations (fusion reactivity, line/synchrotron
# radiation) singular at the edge and stiffens the confinement solve.
_EDGE_PEDESTAL = 0.02


def _parabolic_profile(average: Any, peaking: Any, rho: Any) -> np.ndarray:
    """Return an ``average * shape`` profile whose volume-average equals ``average``.

    ``average`` may be a scalar (per-point) or a batched ``(N, 1)`` column (the
    popcon grid), in which case the result is ``(N, P)``.  ``peaking`` is assumed
    uniform across the batch (the popcon peaking controls are grid constants).
    """
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho must be a one-dimensional profile grid")
    peak = float(np.asarray(peaking, dtype=float).reshape(-1)[0])
    alpha = _alpha_for_peaking(peak, rho_arr)
    shape = np.maximum(1.0 - rho_arr**2, 0.0) ** alpha
    shape = shape * (1.0 - _EDGE_PEDESTAL) + _EDGE_PEDESTAL
    mean = float(volume_average(shape, rho_arr))
    unit = shape / max(mean, 1e-300)
    avg_arr = np.asarray(average, dtype=float)
    if avg_arr.ndim == 0:
        return float(avg_arr) * unit
    return avg_arr.reshape(avg_arr.shape[0], 1) * unit


@relation(
    name="Parabolic ion temperature profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="T_i",
    dependency="generated_profile",
)
def parabolic_ion_temperature_profile(T_i_avg: float, ion_temperature_peaking: float, rho: Any) -> Any:
    """Generate an ion-temperature profile from average and peaking factor."""
    return _parabolic_profile(T_i_avg, ion_temperature_peaking, rho)


@relation(
    name="Parabolic electron temperature profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="T_e",
    dependency="generated_profile",
)
def parabolic_electron_temperature_profile(T_e_avg: float, temperature_peaking: float, rho: Any) -> Any:
    """Generate an electron-temperature profile from average and peaking factor."""
    return _parabolic_profile(T_e_avg, temperature_peaking, rho)


@relation(
    name="Parabolic ion density profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="n_i",
    dependency="generated_profile",
)
def parabolic_ion_density_profile(n_i_avg: float, ion_density_peaking: float, rho: Any) -> Any:
    """Generate an ion-density profile from average and peaking factor."""
    return _parabolic_profile(n_i_avg, ion_density_peaking, rho)


@relation(
    name="Parabolic electron density profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="n_e",
    dependency="generated_profile",
)
def parabolic_electron_density_profile(n_e_avg: float, density_peaking: float, rho: Any) -> Any:
    """Generate an electron-density profile from average and peaking factor."""
    return _parabolic_profile(n_e_avg, density_peaking, rho)


@relation(
    name="Peak temperatures from average and peaking",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs=("T0", "T_i_peak"),
)
def calc_temperature_peaking(
    T_e_avg: float, T_i_avg: float, temperature_peaking: float, ion_temperature_peaking: float
) -> tuple[float, float]:
    """Apply the temperature peaking to obtain on-axis (peak) temperatures.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    fusdb allows the ion temperature profile to peak independently of the
    electron profile; ``ion_temperature_peaking`` defaults to
    ``temperature_peaking`` (the electron value), which recovers cfspopcon's
    single shared peaking.

    Args:
        T_e_avg: :term:`glossary link<average_electron_temp>`
        T_i_avg: :term:`glossary link<average_ion_temp>`
        temperature_peaking: :term:`glossary link<temperature_peaking>`
        ion_temperature_peaking: :term:`glossary link<ion_temperature_peaking>`

    Returns:
        peak_electron_temp (T0), peak_ion_temp (T_i_peak)
    """
    # CHECK
    peak_electron_temp = T_e_avg * temperature_peaking
    peak_ion_temp = T_i_avg * ion_temperature_peaking
    return peak_electron_temp, peak_ion_temp
