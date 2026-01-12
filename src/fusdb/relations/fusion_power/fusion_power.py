"""Fusion power utilities using Bosch-Hale reactivities."""

from numpy import float64

from fusdb.registry.constants import DT_REACTION_ENERGY_J, MEV_TO_J
from fusdb.reactor_class import Reactor
from fusdb.relations.fusion_power.reactivity_functions import sigmav_DT_BoschHale

# NOTE: TEMPORARY! This relation is to be replaced by a more general fusion power relation including different fuel mixtures!
@Reactor.relation(
    "power_exhaust",
    name="DT fusion power (Bosch-Hale)",
    output="P_fus",
    solve_for=("P_fus",),
    constraints=("n_avg >= 0", "T_avg >= 0", "V_p >= 0"),
    rel_tol=0.1
)
def fusion_power_DT_BoschHale(n_avg: float64, T_avg: float64, V_p: float64) -> float64:
    """Return D-T fusion power from average density (m^-3), temperature (keV), and volume (m^3).
    Assuming a 50/50 D-T mix with n_avg ~= n_i = n_D + n_T.
    """
    reactivity = sigmav_DT_BoschHale(T_avg)
    reaction_rate = 0.25 * n_avg ** 2 * reactivity
    return reaction_rate * DT_REACTION_ENERGY_J * V_p
