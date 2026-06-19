"""T-He3 fusion-power relations."""

from typing import Any

from fusdb import relation
from fusdb.registry import (
    THE3_D_ALPHA_ENERGY_J,
    THE3_D_D_ENERGY_J,
    THE3_NP_ALPHA_ENERGY_J,
    THE3_NP_N_ENERGY_J,
    THE3_NP_P_ENERGY_J,
)


@relation(
    name='T-He3 fusion power',
    tags=('fusion_power',),
    outputs='P_fus_THe3',
)
def fusion_power_the3(P_fus_THe3_D: float, P_fus_THe3_np: float) -> Any:
    """Return total T-He3 fusion power from the D and n+p branches."""
    return P_fus_THe3_D + P_fus_THe3_np


@relation(
    name='T-He3 alpha+D fusion power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_D',
)
def fusion_power_the3_d(P_fus_THe3_D_alpha: float, P_fus_THe3_D_D: float) -> Any:
    """Return T-He3 alpha+D branch power from alpha and deuteron components."""
    return P_fus_THe3_D_alpha + P_fus_THe3_D_D


@relation(
    name='T-He3 alpha+n+p fusion power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np',
)
def fusion_power_the3_np(
    P_fus_THe3_np_alpha: float,
    P_fus_THe3_np_n: float,
    P_fus_THe3_np_p: float,
) -> Any:
    """Return T-He3 alpha+n+p branch power from product components."""
    return P_fus_THe3_np_alpha + P_fus_THe3_np_n + P_fus_THe3_np_p


@relation(
    name='T-He3 alpha+D alpha power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_D_alpha',
)
def alpha_power_the3_d(Rr_THe3_D: float) -> Any:
    """Return alpha power from the T-He3 alpha+D branch."""
    return THE3_D_ALPHA_ENERGY_J * Rr_THe3_D


@relation(
    name='T-He3 alpha+D deuteron power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_D_D',
)
def deuteron_power_the3_d(Rr_THe3_D: float) -> Any:
    """Return deuteron power from the T-He3 alpha+D branch."""
    return THE3_D_D_ENERGY_J * Rr_THe3_D


@relation(
    name='T-He3 alpha+n+p alpha power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np_alpha',
)
def alpha_power_the3_np(Rr_THe3_np: float) -> Any:
    """Return alpha power from the T-He3 alpha+n+p branch."""
    return THE3_NP_ALPHA_ENERGY_J * Rr_THe3_np


@relation(
    name='T-He3 alpha+n+p neutron power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np_n',
)
def neutron_power_the3_np(Rr_THe3_np: float) -> Any:
    """Return neutron power from the T-He3 alpha+n+p branch."""
    return THE3_NP_N_ENERGY_J * Rr_THe3_np


@relation(
    name='T-He3 alpha+n+p proton power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np_p',
)
def proton_power_the3_np(Rr_THe3_np: float) -> Any:
    """Return proton power from the T-He3 alpha+n+p branch."""
    return THE3_NP_P_ENERGY_J * Rr_THe3_np
