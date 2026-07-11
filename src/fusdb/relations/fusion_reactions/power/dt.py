"""DT fusion-power relations."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry import (
    DT_ALPHA_ENERGY_J,
    DT_N_ENERGY_J,
)


@relation(
    name='DT fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DT',
)
def fusion_power_dt(P_fus_DT_alpha: float, P_fus_DT_n: float) -> Any:
    """Return total D-T fusion power from alpha and neutron components."""
    return P_fus_DT_alpha + P_fus_DT_n


@relation(
    name='DT alpha power',
    tags=('fusion_power',),
    outputs='P_fus_DT_alpha',
)
def alpha_power_dt(Rr_DT: float) -> Any:
    """Return alpha power from D-T fusion."""
    return DT_ALPHA_ENERGY_J * Rr_DT


@relation(
    name='DT neutron power',
    tags=('fusion_power',),
    outputs='P_fus_DT_n',
)
def neutron_power_dt(Rr_DT: float) -> Any:
    """Return neutron power from D-T fusion."""
    return DT_N_ENERGY_J * Rr_DT
