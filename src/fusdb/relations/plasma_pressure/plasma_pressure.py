"""Plasma pressure relations expressed via @relation decorators."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from fusdb import relation
from fusdb.registry import KEV_TO_J


@relation(
    name='Thermal pressure',
    tags=('plasma',),
    outputs='p_th',
)
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float, rho: float) -> Any:
    """Return volume-averaged thermal pressure from profile/local quantities.

    Args:
        n_e: Electron density profile.
        T_e: Electron temperature profile.
        n_i: Ion density profile.
        T_i: Ion temperature profile.
        rho: Radial grid.

    Returns:
        Volume-averaged thermal pressure.
    """
    integrand = n_e * T_e + n_i * T_i
    return KEV_TO_J * trapezoid(integrand, x=rho)


########################################
@relation(
    name='Thermal stored energy',
    tags=('plasma',),
    outputs='W_th',
)
def thermal_stored_energy(p_th: float, V_p: float) -> float:
    """Return thermal stored energy from pressure and plasma volume.

    Args:
        p_th: Volume-averaged thermal pressure.
        V_p: Plasma volume.

    Returns:
        Thermal stored energy.
    """
    return 1.5 * p_th * V_p


########################################
@relation(
    name='Peak pressure',
    tags=('plasma',),
    outputs='p_peak',
)
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> Any:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
