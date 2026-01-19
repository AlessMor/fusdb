"""Fusion gains and triple products relations.
General efficiency metrics"""

from __future__ import annotations
import numpy as np

from fusdb.reactor_class import Reactor


@Reactor.relation(
    "fusion_power",
    name="Fusion triple product",
    output="n_i_tau_E_T_i",
)
def fusion_triple_product(
    n_i_peak: float, T_i_peak: float, tau_E: float
) -> float:
    """Return fusion triple product from peak ion density, peak ion temperature, and energy confinement time.

    The fusion triple product (n_i * T_i * tau_E) is a key figure of merit for fusion performance,
    representing the combined effect of density, temperature, and confinement.

    Args:
        n_i_peak: Peak fuel ion density [m^-3]
        T_i_peak: Peak ion temperature [keV]
        tau_E: Energy confinement time [s]

    Returns:
        n_i_tau_E_T_i: Fusion triple product [m^-3 keV s]
    """
    return n_i_peak * T_i_peak * tau_E

@Reactor.relation(
    "fusion_power",
    name="Physics gain factor",
    output="Q_sci",
)
def physics_gain_factor(
    P_fus: float, P_aux: float) -> float:
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
    if P_aux == 0:
        Q_phy = np.inf
    else:
        Q_phy = P_fus / P_aux
    return Q_phy

@Reactor.relation(
    "fusion_power",
    name="Engineering gain factor",
    output="Q_eng",
)
def engineering_gain_factor(
    P_fus_el: float, P_aux_el: float) -> float:
    """Return the engineering gain factor Q_eng = (P_fus_el / P_aux_el).

    The engineering gain factor is a measure of the net electric power output relative to the electrical power required to drive external heating sources.
    """
    if P_aux_el == 0 and P_fus_el > 0:
        Q_eng = np.inf
    else:
        Q_eng = (P_fus_el - P_aux_el) / P_aux_el
    return Q_eng

# TODO(med): consider adding the definition used in cfspopcon: Q = P_fusion / (P_ohmic + P_auxiliary_launched)