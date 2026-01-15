"""Power balance relations for loss power."""

from __future__ import annotations

from fusdb.reactor_class import Reactor

# NOTE: CHECK THE DEFINITIONS FOR P_LOSS AND ADD REFERENCES
@Reactor.relation(
    ("power_exhaust", "power_balance"),
    name="Total plasma heating",
    output="P_heating",
)
def total_plasma_heating(P_ohmic: float, P_charged: float, P_aux: float) -> float:
    """Return total plasma heating from ohmic, charged fusion, and auxiliary sources."""
    return P_ohmic + P_charged + P_aux


@Reactor.relation(
    ("power_exhaust", "power_balance"),
    name="Loss power to SOL and core radiation",
    output="P_loss",
)
def loss_power_to_exhaust(P_sep: float, P_rad: float) -> float:
    """Return loss power as the sum of separatrix power and core radiation."""
    return P_sep + P_rad


@Reactor.relation(
    ("power_exhaust", "power_balance"),
    name="Charged fusion power",
    output="P_charged",
)
def charged_fusion_power(
    P_fus_DT_alpha: float,
    P_fus_DDn_He3: float,
    P_fus_DDp_T: float,
    P_fus_DDp_p: float,
    P_fus_DHe3_alpha: float,
    P_fus_DHe3_p: float,
) -> float:
    """Return charged fusion power from common D-T, D-D, and D-He3 channels."""
    return (
        P_fus_DT_alpha
        + P_fus_DDn_He3
        + P_fus_DDp_T
        + P_fus_DDp_p
        + P_fus_DHe3_alpha
        + P_fus_DHe3_p
    )
