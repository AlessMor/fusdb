"""Stored energy and confinement relations expressed via Reactor.relation."""

from __future__ import annotations

from fusdb.registry.constants import KEV_TO_J
from fusdb.reactor_class import Reactor
from fusdb.relation_util import nonzero, symbol


@Reactor.relation("plasma", name="Thermal pressure", output="p_th")
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float) -> float:
    """Return thermal pressure from electron/ion densities and temperatures."""
    return (n_e * T_e + n_i * T_i) * KEV_TO_J


@Reactor.relation("plasma", name="Thermal stored energy", output="W_th", variables=("p_th", "V_p"))
def thermal_stored_energy(p_th: float, V_p: float) -> float:
    """Return thermal stored energy from pressure and volume."""
    return 1.5 * p_th * V_p


@Reactor.relation(
    "plasma",
    name="Energy confinement time",
    output="tau_E",
    variables=("W_th", "P_loss"),
    constraints=(nonzero(symbol("P_loss")),),
)
def energy_confinement_time(W_th: float, P_loss: float) -> float:
    """Return energy confinement time from stored energy and loss power."""
    return W_th / P_loss


THERMAL_PRESSURE_REL = thermal_pressure.relation
THERMAL_ENERGY_REL = thermal_stored_energy.relation
CONFINEMENT_REL = energy_confinement_time.relation
