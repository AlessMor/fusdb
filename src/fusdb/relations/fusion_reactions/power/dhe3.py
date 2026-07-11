"""D-He3 fusion-power relations."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry import (
    DHE3_ALPHA_ENERGY_J,
    DHE3_P_ENERGY_J,
)


@relation(
    name='D-He3 fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DHe3',
)
def fusion_power_dhe3(P_fus_DHe3_alpha: float, P_fus_DHe3_p: float) -> Any:
    """Return total D-He3 fusion power from alpha and proton components."""
    return P_fus_DHe3_alpha + P_fus_DHe3_p


@relation(
    name='D-He3 alpha power',
    tags=('fusion_power',),
    outputs='P_fus_DHe3_alpha',
)
def alpha_power_dhe3(Rr_DHe3: float) -> Any:
    """Return alpha power from D-He3 fusion."""
    return DHE3_ALPHA_ENERGY_J * Rr_DHe3


@relation(
    name='D-He3 proton power',
    tags=('fusion_power',),
    outputs='P_fus_DHe3_p',
)
def proton_power_dhe3(Rr_DHe3: float) -> Any:
    """Return proton power from D-He3 fusion."""
    return DHE3_P_ENERGY_J * Rr_DHe3
