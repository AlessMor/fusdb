"""Bremsstrahlung radiation relations."""

from typing import Any

from fusdb.utils import volume_average

from fusdb.relation import relation


@relation(
    name='Bremsstrahlung radiation',
    tags=('power_balance',),
    outputs='P_brem',
)
def bremsstrahlung_radiation(
    n_e: float,
    T_e: float,
    Z_eff: float,
    V_p: float,
    rho: float,
    w_V: Any = None,
) -> Any:
    """Return total bremsstrahlung radiated power from an explicit local profile law.

    ``rho`` is the computational sampling grid. ``w_V`` supplies the physical
    volume measure when available; omitting it retains the historical
    self-similar weighting.
    """
    n_e20 = n_e / 1e20
    Tm = 511.0
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + (2.0 / Z_eff) * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * Z_eff * (n_e20 ** 2) * (T_e ** 0.5) * xrel * 1e6
    return V_p * volume_average(p_brem, rho, weight=w_V)


@relation(
    name='Hydrogenic bremsstrahlung (cfspopcon)',
    tags=('power_balance',),
    outputs='P_brem',
)
def hydrogenic_bremsstrahlung_cfspopcon(
    n_e: float, T_e: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Bremsstrahlung from the hydrogenic (fuel) plasma only, i.e. at Z_eff = 1.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Same Stott 2005 formula as :func:`bremsstrahlung_radiation`, evaluated at
    ``Z_eff = 1``. cfspopcon's impurity cooling curves already contain impurity
    bremsstrahlung and recombination continuum, so this relation is the paired
    hydrogenic contribution when those curves are active.
    """
    # CHECK
    n_e20 = n_e / 1e20
    Tm = 511.0
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + 2.0 * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * (n_e20 ** 2) * (T_e ** 0.5) * xrel * 1e6
    return V_p * volume_average(p_brem, rho, weight=w_V)


@relation(
    name="Impurity bremsstrahlung from total and hydrogenic",
    tags=("power_balance",),
    outputs="P_brem_imp",
)
def impurity_bremsstrahlung(
    n_e: Any, T_e: Any, T_e_avg: Any, V_p: Any, rho: Any,
    c_Xe: Any = 0.0, c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0,
    c_N: Any = 0.0, c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0,
    c_W: Any = 0.0, w_V: Any = None,
) -> Any:
    """Bremsstrahlung radiated by the impurities alone [W].

    A SUM over impurity species, never a difference of two totals. The local law
    is linear in charge after separating the hydrogenic part, which keeps this
    quantity non-negative by construction.
    """
    # CHECK
    from ..composition.impurities import _mavrin_charge_terms

    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    change_in_zeff = 0.0
    for concentration, zbar in _mavrin_charge_terms(T_e_avg, concentrations):
        change_in_zeff = change_in_zeff + concentration * zbar * (zbar - 1.0)

    n_e20 = n_e / 1e20
    Tm = 511.0
    p_brem_imp = 5.35e-3 * change_in_zeff * (n_e20 ** 2) * (T_e ** 0.5) * (1.0 + 2.0 * T_e / Tm) * 1e6
    return V_p * volume_average(p_brem_imp, rho, weight=w_V)


@relation(
    name="Line radiation from impurity cooling rate",
    tags=("power_balance",),
    outputs="P_line",
)
def line_radiation_from_cooling_rate(P_cool_imp: float, P_brem_imp: float) -> Any:
    """Line (plus recombination) radiated power [W].

    Cooling-rate tables include the impurities' own bremsstrahlung. Subtract it
    here when ``P_brem`` is defined as the total bremsstrahlung so
    ``P_rad = P_brem + P_line + P_sync`` does not double-count it.
    """
    # CHECK
    return P_cool_imp - P_brem_imp


@relation(
    name="Line radiation equals impurity cooling rate (cfspopcon convention)",
    tags=("power_balance",),
    outputs="P_line",
)
def line_radiation_equals_cooling_rate(P_cool_imp: float) -> Any:
    """``P_line = P_cool_imp`` -- cfspopcon's radiation decomposition.

    This convention is paired with hydrogenic-only ``P_brem``; impurity
    bremsstrahlung remains inside the cooling-curve contribution.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_cool_imp
