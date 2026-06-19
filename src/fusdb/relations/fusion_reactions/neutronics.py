"""Neutronics relations."""

from typing import Any

from fusdb import relation


@relation(
    name="Neutron wall loading",
    tags=("fusion_power", "neutronics"),
    outputs="q_wall",
)
def neutron_wall_loading(P_neutron: float, A_p: float) -> Any:
    """Return neutron wall loading from neutron power and plasma surface area."""
    # TODO: This considers plasma surface; it should consider wall surface.
    return (P_neutron / A_p)
