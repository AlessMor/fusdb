"""Separatrix power and power-exhaust relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name="Power crossing the separatrix",
    tags=("power_exhaust",),
    outputs="P_sep",
)
def calc_power_crossing_separatrix(P_in: Any, P_rad: Any) -> Any:
    """Calculate the power crossing the separatrix.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return np.maximum(P_in - P_rad, 0.0)


@relation(
    name="Separatrix power from power balance",
    tags=("power_exhaust", "process"),
    outputs="P_sep",
)
def calculate_separatrix_power(
    f_p_alpha_plasma_deposited: Any,
    P_alpha_total: Any,
    P_non_alpha_charged: Any,
    P_aux: Any,
    P_ohmic: Any,
    P_rad: Any,
) -> Any:
    """Calculate the power crossing the separatrix (P_sep) from the plasma
    power balance.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS evaluates this sum in MW; it is linear (scale-invariant), so fusdb
    evaluates it directly in W.

    Parameters
    ----------
    f_p_alpha_plasma_deposited :
        Fraction of alpha power deposited in plasma
    P_alpha_total :
        Total alpha power produced [W] (PROCESS p_alpha_total_mw)
    P_non_alpha_charged :
        Power from non-alpha charged particles [W] (PROCESS p_non_alpha_charged_mw)
    P_aux :
        Total power injected by heating and current drive [W]
        (PROCESS p_hcd_injected_total_mw)
    P_ohmic :
        Ohmic heating power [W] (PROCESS p_plasma_ohmic_mw)
    P_rad :
        Radiated power from plasma [W] (PROCESS p_plasma_rad_mw)

    Returns
    -------
    :
        Power crossing the separatrix [W]
    """
    # CHECK
    return (
        f_p_alpha_plasma_deposited * P_alpha_total
        + P_non_alpha_charged
        + P_aux
        + P_ohmic
        - P_rad
    )


@relation(
    name="L-H transition threshold power (Martin-Ryter)",
    tags=("power_exhaust", "tokamak"),
    outputs="P_LH",
)
def calc_LH_transition_threshold_power(
    I_p: Any,
    B0: Any,
    a: Any,
    R: Any,
    A_p: Any,
    afuel: Any,
    n_e_avg: Any,
    confinement_threshold_scalar: Any = 1.0,
) -> Any:
    """Calculate the threshold power (crossing the separatrix) to transition into H-mode.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    From Martin NF 2008 scaling, with mass correction :cite:`martin_power_2008`, plus the
    low-density branch from Ryter 2014 :cite:`Ryter_2014`. Full Martin+Ryter form; fusdb's
    default "L-H transition threshold power" is a simpler Martin-style scaling.

    Args:
        I_p: [A] :term:`glossary link<plasma_current>`
        B0: [T] :term:`glossary link<magnetic_field_on_axis>`
        a: [m] :term:`glossary link<minor_radius>`
        R: [m] :term:`glossary link<major_radius>`
        A_p: [m^2] :term:`glossary link<surface_area>`
        afuel: [amu] :term:`glossary link<average_ion_mass>`
        n_e_avg: [1/m^3] :term:`glossary link<average_electron_density>`
        confinement_threshold_scalar: [~] :term:`glossary link<confinement_threshold_scalar>`

    Returns:
        P_LH [W]
    """
    # CHECK
    plasma_current = I_p / 1.0e6  # cfspopcon uses MA
    n19 = n_e_avg / 1.0e19  # cfspopcon uses 1e19 m^-3

    def _calc_Martin_LH_threshold(electron_density: Any) -> Any:
        _DEUTERIUM_MASS_NUMBER = 2.0
        return (0.0488 * ((electron_density / 10.0) ** 0.717) * (B0**0.803) * (A_p**0.941)) * (
            _DEUTERIUM_MASS_NUMBER / afuel
        )

    # Ryter 2014, equation 3 (low-density rollover)
    neMin19 = 0.7 * (plasma_current**0.34) * (B0**0.62) * (a**-0.95) * ((R / a) ** 0.4)

    if n19 < neMin19:
        P_LH_thresh = _calc_Martin_LH_threshold(electron_density=neMin19)
        return 1.0e6 * (P_LH_thresh * (neMin19 / n19) ** 2.0) * confinement_threshold_scalar
    P_LH_thresh = _calc_Martin_LH_threshold(electron_density=n19)
    return 1.0e6 * P_LH_thresh * confinement_threshold_scalar


@relation(
    name="Ratio of P_SOL to P_LH",
    tags=("power_exhaust", "tokamak"),
    outputs="ratio_of_P_SOL_to_P_LH",
)
def calc_ratio_P_LH(P_sep: Any, P_LH: Any) -> Any:
    """Ratio of the power crossing the separatrix to the L-H threshold power.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_sep / P_LH


@relation(
    name='P_sep ratio',
    tags=('power_exhaust',),

    outputs='P_sep_over_R',
)
def p_sep_ratio(P_sep: float, R: float) -> Any:
    """Return the P_sep / R ratio."""
    return P_sep / R


@relation(
    name='P_sep metric',
    tags=('power_exhaust',),

    outputs='P_sep_B_over_q95AR',
)
def p_sep_metric(P_sep: float, B0: float, q95: float, A: float, R: float) -> Any:
    """Return the P_sep * B0 / (q95 * A * R) metric."""
    return P_sep * B0 / (q95 * A * R)
