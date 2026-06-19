"""DD fusion-power relations."""

from typing import Any

from fusdb import relation
from fusdb.registry import (
    DD_HE3_ENERGY_J,
    DD_N_ENERGY_J,
    DD_P_ENERGY_J,
    DD_T_ENERGY_J,
)


@relation(
    name='DD fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DD',
)
def fusion_power_dd(P_fus_DDn: float, P_fus_DDp: float) -> Any:
    """Return total D-D fusion power from both branches."""
    return P_fus_DDn + P_fus_DDp


@relation(
    name='DD (T+p) fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DDp',
)
def fusion_power_ddp(P_fus_DDp_T: float, P_fus_DDp_p: float) -> Any:
    """Return D-D fusion power from the T+p branch."""
    return P_fus_DDp_T + P_fus_DDp_p


@relation(
    name='DD (He3+n) fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DDn',
)
def fusion_power_ddn(P_fus_DDn_He3: float, P_fus_DDn_n: float) -> Any:
    """Return D-D fusion power from the He3+n branch."""
    return P_fus_DDn_He3 + P_fus_DDn_n


@relation(
    name='DDp triton power',
    tags=('fusion_power',),
    outputs='P_fus_DDp_T',
)
def triton_power_dd(Rr_DDp: float) -> Any:
    """Return triton power from the D(d,p)T branch."""
    return DD_T_ENERGY_J * Rr_DDp


@relation(
    name='DDp proton power',
    tags=('fusion_power',),
    outputs='P_fus_DDp_p',
)
def proton_power_dd(Rr_DDp: float) -> Any:
    """Return proton power from the D(d,p)T branch."""
    return DD_P_ENERGY_J * Rr_DDp


@relation(
    name='DDn helium-3 power',
    tags=('fusion_power',),
    outputs='P_fus_DDn_He3',
)
def he3_power_dd(Rr_DDn: float) -> Any:
    """Return He3 power from the D(d,n)He3 branch."""
    return DD_HE3_ENERGY_J * Rr_DDn


@relation(
    name='DDn neutron power',
    tags=('fusion_power',),
    outputs='P_fus_DDn_n',
)
def neutron_power_dd(Rr_DDn: float) -> Any:
    """Return neutron power from the D(d,n)He3 branch."""
    return DD_N_ENERGY_J * Rr_DDn
