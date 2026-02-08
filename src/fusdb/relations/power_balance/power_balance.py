"""Power balance relations for loss power."""

from __future__ import annotations

from fusdb.relation_class import Relation_decorator as Relation
# TODO(high): CHECK THE DEFINITIONS FOR P_LOSS AND ADD REFERENCES
@Relation(
    name="Total plasma heating",
    output="P_heating",
    tags=("power_balance",),
)
def total_plasma_heating(P_ohmic: float, P_charged: float, P_aux: float) -> float:
    """Return total plasma heating from ohmic, charged fusion, and auxiliary sources."""
    return P_ohmic + P_charged + P_aux


########################################
@Relation(
    name="Loss power to SOL and core radiation",
    output="P_loss_explicit",
    tags=("power_balance",),
)
def loss_power_to_exhaust(P_sep: float, P_rad: float) -> float:
    """Return loss power as the sum of separatrix power and core radiation."""
    return P_sep + P_rad


########################################
@Relation(
    name="Power balance",
    output="P_loss",
    tags=("power_balance",),
)
def power_balance_simple(P_heating: float) -> float:
    """Total power lost must equal total power input."""
    return P_heating


# TODO(med): from PROCESS, add balance for ions and electrons (constraints.py)
