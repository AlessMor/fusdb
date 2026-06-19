"""TT fusion-power relations."""

from typing import Any

from fusdb import relation
from fusdb.registry import (
    TT_ALPHA_ENERGY_J,
    TT_N_ENERGY_J,
)


@relation(
    name='TT fusion power',
    tags=('fusion_power',),
    outputs='P_fus_TT',
)
def fusion_power_tt(P_fus_TT_alpha: float, P_fus_TT_n: float) -> Any:
    """Return total T-T fusion power from alpha and neutron components."""
    return P_fus_TT_alpha + P_fus_TT_n


@relation(
    name='TT alpha power',
    tags=('fusion_power',),
    outputs='P_fus_TT_alpha',
)
def alpha_power_tt(Rr_TT: float) -> Any:
    """Return alpha power from T-T fusion."""
    return TT_ALPHA_ENERGY_J * Rr_TT


@relation(
    name='TT neutron power',
    tags=('fusion_power',),
    outputs='P_fus_TT_n',
)
def neutron_power_tt(Rr_TT: float) -> Any:
    """Return combined neutron power from T-T fusion."""
    return TT_N_ENERGY_J * Rr_TT
