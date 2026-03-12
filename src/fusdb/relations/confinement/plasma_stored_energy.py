"""Stored energy and confinement relations expressed via @relation decorators."""

from __future__ import annotations

from fusdb.relation_util import relation
@relation(
    name="Thermal stored energy",
    output="W_th",
    tags=("plasma",),
)
def thermal_stored_energy(p_th: float, V_p: float) -> float:
    """Return thermal stored energy from pressure and volume."""
    return 1.5 * p_th * V_p
########################################
@relation(
    name="Energy confinement time",
    output="tau_E",
    tags=("plasma",),
    constraints=("P_loss != 0",),
)
def energy_confinement_time(W_th: float, P_loss: float) -> float:
    """Return energy confinement time from stored energy and loss power."""
    return W_th / P_loss
