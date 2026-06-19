"""He3-He3 fusion-power relations."""

from typing import Any

from fusdb import relation
from fusdb.registry import (
    HE3HE3_ALPHA_ENERGY_J,
    HE3HE3_P_ENERGY_J,
)


@relation(
    name='He3-He3 fusion power',
    tags=('fusion_power',),
    outputs='P_fus_He3He3',
)
def fusion_power_he3he3(P_fus_He3He3_alpha: float, P_fus_He3He3_p: float) -> Any:
    """Return total He3-He3 fusion power from alpha and proton components.

    Args:
        P_fus_He3He3_alpha: Alpha power from He3-He3 fusion.
        P_fus_He3He3_p: Combined proton power from He3-He3 fusion.

    Returns:
        Total He3-He3 fusion power.
    """
    return P_fus_He3He3_alpha + P_fus_He3He3_p


@relation(
    name='He3-He3 alpha power',
    tags=('fusion_power',),
    outputs='P_fus_He3He3_alpha',
)
def alpha_power_he3he3(Rr_He3He3: float) -> Any:
    """Return alpha power from He3-He3 fusion.

    Args:
        Rr_He3He3: He3-He3 reaction rate.

    Returns:
        Alpha power from He3-He3 fusion.
    """
    return HE3HE3_ALPHA_ENERGY_J * Rr_He3He3


@relation(
    name='He3-He3 proton power',
    tags=('fusion_power',),
    outputs='P_fus_He3He3_p',
)
def proton_power_he3he3(Rr_He3He3: float) -> Any:
    """Return combined proton power from He3-He3 fusion.

    Args:
        Rr_He3He3: He3-He3 reaction rate.

    Returns:
        Combined power from both He3-He3 protons.
    """
    return HE3HE3_P_ENERGY_J * Rr_He3He3
