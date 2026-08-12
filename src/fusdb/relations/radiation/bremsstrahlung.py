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
    n_e: float, T_e: float, Z_eff: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return total bremsstrahlung radiated power from an explicit local profile law.

    Args:
        n_e: Electron density [1/m^3].
        T_e: Electron temperature [keV].
        Z_eff: Effective charge [dimensionless].
        V_p: Plasma volume [m^3].
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Return:
        Total bremsstrahlung radiated power [W].
    """
    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, electron rest mass energy
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + (2.0 / Z_eff) * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * Z_eff * (n_e20 ** 2) * (T_e ** 0.5) * xrel * 1e6  # [W/m^3]
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
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Return:
        Hydrogenic bremsstrahlung radiated power [W].
    """
    # CHECK
    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, electron rest mass energy
    xrel = (1.0 + 2.0 * T_e / Tm) * (1.0 + 2.0 * (1.0 - 1.0 / (1.0 + T_e / Tm)))
    p_brem = 5.35e-3 * (n_e20 ** 2) * (T_e ** 0.5) * xrel * 1e6  # [W/m^3]
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

    A SUM over impurity species, never a difference of two totals.

    The local law is linear in Z: expanding its relativistic factor,
    ``p(Z) = C (1 + 2 T_e/Tm) (Z + 2 b)``, so the impurity part is
    ``C (1 + 2 T_e/Tm) (Z_eff - 1)`` exactly.  Quasineutrality then gives

        Z_eff - 1 = sum_z c_z Zbar_z (Zbar_z - 1)

    (cfspopcon's ``change_in_zeff``), and every term of that is >= 0 because
    Zbar >= 1.  So this quantity is NON-NEGATIVE BY CONSTRUCTION, and a
    pure-hydrogenic plasma returns exactly 0.0 because the sum is EMPTY.

    Previously it was ``brems(Z_eff) - brems(1)``.  That is the same number when
    Z_eff is exact, but Z_eff is SOLVED: on ARC_V0 it converges to 0.99999997
    and the difference came out at -0.159 W, tripping the ``[0, inf)`` domain.
    Same structural defect as the old negative-``P_line`` bug, which was a
    subtraction for the same reason.
    """
    # CHECK
    from ..composition.impurities import _mavrin_charge_terms

    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    change_in_zeff = 0.0
    for concentration, zbar in _mavrin_charge_terms(T_e_avg, concentrations):
        change_in_zeff = change_in_zeff + concentration * zbar * (zbar - 1.0)

    n_e20 = n_e / 1e20
    Tm = 511.0  # keV, electron rest mass energy
    p_brem_imp = 5.35e-3 * change_in_zeff * (n_e20 ** 2) * (T_e ** 0.5) * (1.0 + 2.0 * T_e / Tm) * 1e6
    return V_p * volume_average(p_brem_imp, rho, weight=w_V)


@relation(
    name="Line radiation from impurity cooling rate",
    tags=("power_balance",),
    outputs="P_line",
)
def line_radiation_from_cooling_rate(P_cool_imp: float, P_brem_imp: float) -> Any:
    """Line (plus recombination) radiated power [W].

    Cooling-rate tables -- Mavrin, radas, Post-Jensen, PROCESS's own -- return the
    TOTAL impurity radiated power: line, recombination continuum and the
    impurities' own bremsstrahlung. ``P_brem`` is meanwhile defined as the TOTAL
    bremsstrahlung, hydrogenic and impurity alike. So the impurity bremsstrahlung
    appears in both, and summing them into ``P_rad`` would count it twice -- ~16%
    of the radiated power at reactor conditions.

    Removing it here is what keeps each variable meaning what its name says and
    leaves ``P_rad = P_brem + P_line + P_sync`` self-consistent. Recombination
    stays inside ``P_line`` because the tables do not separate it from line
    emission.
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

    cfspopcon does not split the impurities' own bremsstrahlung out of its L_z
    curves at all: it pairs the L_z total with a HYDROGENIC-only ``P_brem``, so
    the impurity bremsstrahlung is carried inside what it calls the impurity
    radiation.  That is self-consistent -- ``P_rad`` comes out right -- but it
    means its ``P_brem`` is not the total bremsstrahlung and its impurity term is
    not line radiation alone.

    Reproducing cfspopcon therefore needs this identity *together with* its
    hydrogenic ``P_brem``; the two go as a pair.  fusdb's default instead derives
    ``P_line`` properly (see ``Line radiation from impurity cooling rate``) so
    that ``P_brem`` can mean the total bremsstrahlung.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_cool_imp
