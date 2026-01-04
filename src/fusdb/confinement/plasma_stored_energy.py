"""Stored energy and confinement relations expressed via Reactor.relation."""

from __future__ import annotations

from fusdb.reactors_class import Reactor
from fusdb.relations_util import require_nonzero

KEV_TO_J = 1.602176634e-16  # conversion factor for temperatures stored in keV


@Reactor.relation("plasma", name="Thermal pressure", output="p_th")
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float) -> float:
    return (n_e * T_e + n_i * T_i) * KEV_TO_J


@Reactor.relation("plasma", name="Thermal stored energy", output="W_th", variables=("p_th", "V_p"))
def thermal_stored_energy(p_th: float, V_p: float) -> float:
    return 1.5 * p_th * V_p


@Reactor.relation("plasma", name="Energy confinement time", output="tau_E", variables=("W_th", "P_loss"))
def energy_confinement_time(W_th: float, P_loss: float) -> float:
    require_nonzero(P_loss, "P_loss", "confinement relation")
    return W_th / P_loss


THERMAL_PRESSURE_REL = thermal_pressure.relation
THERMAL_ENERGY_REL = thermal_stored_energy.relation
CONFINEMENT_REL = energy_confinement_time.relation

__all__ = [
    "KEV_TO_J",
    "THERMAL_PRESSURE_REL",
    "THERMAL_ENERGY_REL",
    "CONFINEMENT_REL",
]
