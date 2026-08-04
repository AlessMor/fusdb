"""Fusion gain and triple-product relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name='Fusion triple product',
    tags=('fusion_power',),
    outputs='n_i_tau_E_T_i',
)
def fusion_triple_product(
    n_i_peak: float, T_i_peak: float, tau_E: float
) -> Any:
    """Return fusion triple product from peak ion density, peak ion temperature, and energy confinement time.
    # TODO: check why peak values are used instead of volume-averaged values
    Args:
        n_i_peak: Peak fuel ion density [m^-3]
        T_i_peak: Peak ion temperature [keV]
        tau_E: Energy confinement time [s]

    Returns:
        n_i_tau_E_T_i: Fusion triple product [m^-3 keV s]
    """
    return n_i_peak * T_i_peak * tau_E


@relation(
    name='Physics gain factor',
    tags=('fusion_power',),
    outputs='Q_sci',
)
def physics_gain_factor(
    P_fus: float, P_aux: float) -> Any:
    """Return the physics gain factor Q_phy = (P_fus / P_aux) for fusion power.

    The physics gain factor is a measure of the fusion power output relative to the ABSORBED auxiliary power.
    - Scientifc Breakeven: Q_phy = 1 
    - Burning plasma: Q_phy >= 5 (at Q = 5 P_alpha = P_aux in DT fusion)
    - Ignition: Q_phy -> infinity (P_aux = 0)
    Args:
        P_fus: Fusion power [W]
        P_aux: Auxiliary power [W]

    Returns:
        physics_gain_factor: Physics gain factor [dimensionless]
    """
    # Q_sci has domain [0, inf): a negative value would mean P_aux < 0 (the
    # plasma is past ignition and would need power *removed* to stay in steady
    # state). Floor at the domain lower limit rather than emit a negative gain.
    return np.maximum(P_fus / P_aux, 0.0)


@relation(
    name='Fusion gain (cfspopcon)',
    tags=('fusion_power',),
    outputs='Q_cfspopcon',
)
def cfspopcon_gain_factor(
    P_fus: float, P_aux_launched: float, P_ohmic: float) -> Any:
    """Return cfspopcon's fusion gain Q = P_fus / (P_aux_launched + P_ohmic).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    cfspopcon divides the fusion power by the *launched* external power,
    which counts ohmic heating alongside the launched auxiliary power
    (its dataset Q equals P_fusion / (P_aux_launched + P_ohmic) to 0.000%).
    Note this is NOT the dataset's own ``P_external`` field, which is the
    different quantity ``P_in - P_charged`` and differs from the sum above by
    ~11%; fusdb spells that one ``P_external`` (see auxiliary.py).
    Unlike Q_sci (absorbed-power convention), the denominator keeps the
    ohmic term, so the gain stays finite where P_aux reaches zero.

    Args:
        P_fus: Fusion power [W]
        P_aux_launched: Launched auxiliary heating power [W]
        P_ohmic: Ohmic heating power [W]

    Returns:
        Q_cfspopcon: Fusion gain on cfspopcon's launched-power convention [dimensionless]
    """
    return P_fus / (P_aux_launched + P_ohmic)


@relation(
    name='Engineering gain factor',
    tags=('fusion_power',),
    outputs='Q_eng',
)
def engineering_gain_factor(
    P_e_net: float, P_aux_el: float) -> Any:
    """Return the engineering gain factor Q_eng = (P_fus_el / P_aux_el).

    The engineering gain factor is a measure of the net electric power output relative to the electrical power required to drive external heating sources.
    """
    return (P_e_net - P_aux_el) / P_aux_el
