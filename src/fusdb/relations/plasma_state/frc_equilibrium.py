"""Rigid-rotor field-reversed-configuration equilibrium quantities.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J, MU0


def _solve_k(xs: Any) -> np.ndarray:
    """Solve tanh(K)/K = 1 - xs^2/2 (VSC Eq. 65)."""
    x = np.asarray(xs, dtype=float)
    target = 1.0 - 0.5 * x**2
    k = np.maximum(np.sqrt(np.maximum(3.0 * (1.0 - target), 1.0e-12)), 1.0e-6)
    for _ in range(30):
        th = np.tanh(k)
        sech2 = 1.0 / np.cosh(k) ** 2
        f = th / k - target
        df = (k * sech2 - th) / k**2
        step = np.where(np.abs(df) > 1.0e-14, f / df, 0.0)
        next_k = np.maximum(k - step, 1.0e-8)
        if np.all(np.abs(next_k - k) <= 1.0e-12 * np.maximum(1.0, np.abs(k))):
            k = next_k
            break
        k = next_k
    return k


@relation(name="FRC rigid-rotor parameter", tags=("frc", "plasma"), outputs="K_frc")
def frc_rigid_rotor_parameter(x_s: Any) -> Any:
    return _solve_k(x_s)


@relation(name="FRC field-null peak pressure", tags=("frc", "plasma"), outputs="p_peak_frc")
def frc_field_null_peak_pressure(B_e: Any) -> Any:
    """Magnetic pressure B_e^2/(2 mu0), VSC Eq. (66)."""
    return np.asarray(B_e) ** 2 / (2.0 * MU0)


@relation(name="FRC peak ion density from pressure balance", tags=("frc", "plasma"), outputs="n_i_peak")
def frc_peak_ion_density_from_pressure_balance(p_peak_frc: Any, T_i_peak: Any, T0: Any, zeta_ne_ni: Any) -> Any:
    """VSC Eq. (66): p_m=n_im (T_i + zeta T_e)."""
    energy = (np.asarray(T_i_peak) + np.asarray(zeta_ne_ni) * np.asarray(T0)) * KEV_TO_J
    return np.asarray(p_peak_frc) / energy


@relation(name="FRC peak electron density", tags=("frc", "plasma"), outputs="n0")
def frc_peak_electron_density(n_i_peak: Any, zeta_ne_ni: Any) -> Any:
    return np.asarray(zeta_ne_ni) * np.asarray(n_i_peak)


@relation(name="FRC trapped poloidal flux", tags=("frc", "plasma"), outputs="phi_p")
def frc_trapped_poloidal_flux(B_e: Any, r_s: Any, K_frc: Any) -> Any:
    """VSC Eq. (73)."""
    return np.pi * np.asarray(B_e) * np.asarray(r_s) ** 2 * np.log(np.cosh(np.asarray(K_frc))) / (2.0 * np.asarray(K_frc))
