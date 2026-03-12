"""Bremsstrahlung radiation relations."""

from __future__ import annotations

from fusdb.relation_util import relation
from fusdb.utils import integrate_profile_over_volume


@relation(
    name="Bremsstrahlung radiation",
    output="P_brem",
    tags=("power_balance",),
    constraints=("T_e >= 0", "n_e >= 0", "V_p >= 0", "Z_eff > 0", "P_brem >= 0"),
)
def bremsstrahlung_radiation(n_e: float, T_e: float, Z_eff: float, V_p: float) -> float:
    """Return total bremsstrahlung radiated power from an explicit local profile law.

    Args:
        n_e: Electron density [1/m^3].
        T_e: Electron temperature [keV].
        Z_eff: Effective charge [dimensionless].
        V_p: Plasma volume [m^3].

    Return:
        Total bremsstrahlung radiated power [W].
    """
    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, electron rest mass energy
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + (2.0 / Z_eff) * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * Z_eff * (n_e20 ** 2) * (T_e ** 0.5) * xrel / 1e6  # [W/m^3]
    total = integrate_profile_over_volume(p_brem, V_p)
    if total is None:
        raise ValueError("Cannot integrate bremsstrahlung profile over volume.")
    return total
