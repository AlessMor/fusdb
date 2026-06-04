"""Plasma pressure relations expressed via @relation decorators."""
from __future__ import annotations

from typing import Any

from fusdb_pyomo import relation
from fusdb_pyomo.registry import KEV_TO_J
import numpy as np


@relation(
    name='Thermal pressure',
    tags=('plasma',),
    outputs='p_th',
)
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float, rho: float) -> Any:
    """Return volume-averaged thermal pressure from profile/local quantities."""
    integrand = n_e * T_e + n_i * T_i
    return KEV_TO_J * np.trapezoid(integrand, x=rho)


########################################
@relation(
    name='Peak pressure',
    tags=('plasma',),
    outputs='p_peak',
)
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> Any:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
