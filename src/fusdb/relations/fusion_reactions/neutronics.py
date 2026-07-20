"""Neutronics relations."""

from typing import Any

from fusdb.relation import relation


@relation(
    name="Neutron wall loading",
    tags=("fusion_power", "neutronics"),
    outputs="q_wall",
)
def neutron_wall_loading(P_neutron: float, A_p: float) -> Any:
    """Return neutron wall loading from neutron power and plasma surface area."""
    # TODO: This considers plasma surface; it should consider wall surface.
    return (P_neutron / A_p)


@relation(
    name="Neutron production rate",
    tags=("fusion_power", "neutronics"),
    outputs="neutron_rate",
)
def neutron_production_rate(
    Rr_DT: float,
    Rr_DDn: float,
    Rr_THe3_np: float = 0.0,
    Rr_TT: float = 0.0,
) -> Any:
    """Return the total neutron production rate from every neutron-producing channel.

    Each reaction rate is weighted by the number of neutrons it releases: D-T,
    the D-D (He3+n) branch and the T-He3 (alpha+n+p) branch each emit one
    neutron per reaction, while T-T emits two.  Mirrors the channel set summed
    in ``Neutron fusion power`` (``P_neutron``); the He3-bearing channels are
    optional and contribute zero when their species are absent and the channel
    has been pruned from the system.
    """
    return Rr_DT + Rr_DDn + Rr_THe3_np + 2.0 * Rr_TT
