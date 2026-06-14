"""Bremsstrahlung radiation relations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from fusdb import relation


@relation(
    name='Bremsstrahlung radiation',
    tags=('power_balance',),
    outputs='P_brem',
)
def bremsstrahlung_radiation(n_e: float, T_e: float, Z_eff: float, V_p: float, rho: float) -> Any:
    """Return total bremsstrahlung radiated power from an explicit local profile law.

    Args:
        n_e: Electron density [1/m^3].
        T_e: Electron temperature [keV].
        Z_eff: Effective charge [dimensionless].
        V_p: Plasma volume [m^3].
        rho: Normalized minor-radius grid for profile integration.

    Return:
        Total bremsstrahlung radiated power [W].
    """
    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, electron rest mass energy
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + (2.0 / Z_eff) * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * Z_eff * (n_e20 ** 2) * (T_e ** 0.5) * xrel / 1e6  # [W/m^3]
    return V_p * trapezoid(p_brem, x=rho)
