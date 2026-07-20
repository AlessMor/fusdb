"""Bremsstrahlung radiation relations."""

from typing import Any

from fusdb.utils import volume_average

from fusdb.relation import relation


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
    p_brem = 5.35e-3 * Z_eff * (n_e20 ** 2) * (T_e ** 0.5) * xrel * 1e6  # [W/m^3]
    return V_p * volume_average(p_brem, rho)


@relation(
    name='Hydrogenic bremsstrahlung (cfspopcon)',
    tags=('power_balance',),
    outputs='P_brem',
)
def hydrogenic_bremsstrahlung_cfspopcon(n_e: float, T_e: float, V_p: float, rho: float) -> Any:
    """Bremsstrahlung from the hydrogenic (fuel) plasma only, i.e. at Z_eff = 1.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Same Stott 2005 formula as :func:`bremsstrahlung_radiation`, evaluated at
    ``Z_eff = 1``.  cfspopcon's ``calc_intrinsic_radiated_power_from_core`` sums
    ``P_brem(Z_eff=1) + P_sync + P_line`` whenever an impurity radiation method
    is active, because the impurity cooling curves L_z already include each
    impurity's own bremsstrahlung and recombination continuum; keeping the
    Z_eff-scaled bremsstrahlung alongside them would double-count it.  Select
    this relation whenever ``P_line`` is driven by supplied impurity
    concentrations (the default Z_eff-scaled relation is the standalone form).

    Args:
        n_e: Electron density [1/m^3].
        T_e: Electron temperature [keV].
        V_p: Plasma volume [m^3].
        rho: Normalized minor-radius grid for profile integration.

    Return:
        Hydrogenic bremsstrahlung radiated power [W].
    """
    # CHECK
    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, electron rest mass energy
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + 2.0 * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * (n_e20 ** 2) * (T_e ** 0.5) * xrel * 1e6  # [W/m^3]
    return V_p * volume_average(p_brem, rho)
