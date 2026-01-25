"""Power balance relations for loss power."""

from __future__ import annotations

from fusdb.reactor_class import Reactor

# TODO(high): CHECK THE DEFINITIONS FOR P_LOSS AND ADD REFERENCES
@Reactor.relation(
    "power_balance",
    name="Total plasma heating",
    output="P_heating",
)
def total_plasma_heating(P_ohmic: float, P_charged: float, P_aux: float) -> float:
    """Return total plasma heating from ohmic, charged fusion, and auxiliary sources."""
    return P_ohmic + P_charged + P_aux


@Reactor.relation(
    "power_balance",
    name="Loss power to SOL and core radiation",
    output="P_loss",
)
def loss_power_to_exhaust(P_sep: float, P_rad: float) -> float:
    """Return loss power as the sum of separatrix power and core radiation."""
    return P_sep + P_rad


@Reactor.relation(
    "power_balance",
    name="Power balance",
    output="P_loss",
)
def power_balance_simple(P_heating: float) -> float:
    """Total power lost must equal total power input."""
    return P_heating
# NOTE: P_loss is computed as a variable in the solver to enforce P_loss = P_heating
# TODO(med): improve this enforcing relation... maybe inside constraints?

# TODO(med): from PROCESS, add balance for ions and electrons (constraints.py)