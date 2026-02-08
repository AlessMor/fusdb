"""Bremsstrahlung radiation relations."""

from __future__ import annotations

from fusdb.relation_class import Relation_decorator as Relation
@Relation(
    name="Bremsstrahlung radiation",
    output="P_brem",
    tags=("power_balance",),
    constraints=("T_avg >= 0", "n_e >= 0", "V_p >= 0", "Z_eff > 0", "P_brem >= 0"),
)
def bremsstrahlung_radiation(n_e: float, T_avg: float, Z_eff: float, V_p: float) -> float:
    """Return bremsstrahlung radiated power using the Stott 2005 formulation.

    Args:
        n_e: Electron density [1/m^3].
        T_avg: Volume-averaged electron temperature [keV].
        Z_eff: Effective charge [dimensionless].
        V_p: Plasma volume [m^3].

    Return:
        Total bremsstrahlung radiated power [W].
    """
    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, T_m = m_e * c**2 electron rest mass
    xrel = (1.0 + 2.0 * T_avg / Tm) * (1.0 + (2.0 / Z_eff) * (1.0 - 1.0 / (1.0 + T_avg / Tm)))
    p_brem = 5.35e-3 * Z_eff * (n_e20 ** 2) * (T_avg ** 0.5) * xrel / 1e6  # Radiated bremsstrahlung power per cubic meter [W / m^3]
    P_brem = V_p * p_brem  # Total bremsstrahlung power [W]
    return P_brem
