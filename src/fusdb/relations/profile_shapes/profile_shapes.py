"""Selectable profile-generation relations.

These relations generate canonical profile arrays from scalar averages and shape
parameters.  They are normal FusDB relations; users select one profile provider
per profile variable through tags or relation include/exclude.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from fusdb import relation


@lru_cache(maxsize=16)
def _peaking_table(rho_key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return a monotone ``(peakings, alphas)`` table for one rho grid.

    ``peak/average`` of ``(1-rho^2)^alpha`` is strictly increasing in alpha, so
    one precomputed table per grid inverts the peaking->alpha map in O(log n)
    by interpolation instead of an 80-step bisection on every profile build.
    """
    rho = np.asarray(rho_key, dtype=float)
    base = np.maximum(1.0 - rho**2, 0.0)
    width = float(rho[-1] - rho[0]) if rho.size > 1 else 1.0
    alphas = np.linspace(0.0, 50.0, 2001)
    peaks = np.empty_like(alphas)
    for i, alpha in enumerate(alphas):
        shape = base**alpha
        mean = float(np.mean(shape)) if width <= 0.0 else float(trapezoid(shape, x=rho) / width)
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


def _parabolic_profile(average: Any, peaking: Any, rho: Any) -> np.ndarray:
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho must be a one-dimensional profile grid")
    avg = float(np.asarray(average, dtype=float))
    peak = float(np.asarray(peaking, dtype=float))
    alpha = _alpha_for_peaking(peak, rho_arr)
    shape = np.maximum(1.0 - rho_arr**2, 0.0) ** alpha
    width = float(rho_arr[-1] - rho_arr[0]) if rho_arr.size > 1 else 1.0
    if width <= 0.0:
        mean = float(np.mean(shape))
    else:
        mean = float(trapezoid(shape, x=rho_arr) / width)
    mean = max(mean, 1e-300)
    return avg * shape / mean


@relation(
    name="Parabolic ion temperature profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="T_i",
    dependency="generated_profile",
)
def parabolic_ion_temperature_profile(T_i_avg: float, temperature_peaking: float, rho: Any) -> Any:
    """Generate an ion-temperature profile from average and peaking factor."""
    return _parabolic_profile(T_i_avg, temperature_peaking, rho)


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
def parabolic_ion_density_profile(n_i_avg: float, density_peaking: float, rho: Any) -> Any:
    """Generate an ion-density profile from average and peaking factor."""
    return _parabolic_profile(n_i_avg, density_peaking, rho)


@relation(
    name="Parabolic electron density profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="n_e",
    dependency="generated_profile",
)
def parabolic_electron_density_profile(n_e_avg: float, density_peaking: float, rho: Any) -> Any:
    """Generate an electron-density profile from average and peaking factor."""
    return _parabolic_profile(n_e_avg, density_peaking, rho)


