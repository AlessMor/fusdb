"""Plasma pressure state relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import rho_average, volume_average
from fusdb.registry import KEV_TO_J


def _thermal_pressure_profile(n_e: Any, T_e: Any, n_i: Any, T_i: Any) -> Any:
    """Return the local thermal pressure profile in Pa."""
    return KEV_TO_J * (
        np.asarray(n_e, dtype=float) * np.asarray(T_e, dtype=float)
        + np.asarray(n_i, dtype=float) * np.asarray(T_i, dtype=float)
    )


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
    return volume_average(_thermal_pressure_profile(n_e, T_e, n_i, T_i), rho)


@relation(
    name='Thermal pressure rho-average',
    tags=('plasma',),
    outputs='p_th_rho_avg',
)
def thermal_pressure_rho_average(n_e: float, T_e: float, n_i: float, T_i: float, rho: float) -> Any:
    """Return straight-rho averaged thermal pressure from profile/local quantities."""
    return rho_average(_thermal_pressure_profile(n_e, T_e, n_i, T_i), rho)


@relation(
    name='Peak pressure',
    tags=('plasma',),
    outputs='p_peak',
)
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> Any:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
