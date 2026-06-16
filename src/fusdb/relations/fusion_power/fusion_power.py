"""
The most favourable fusion reactions (up to 3He):
(DT)    D + T → α (3.52 MeV) + n (14.08 MeV)                    Q = 17.6 MeV
(DDp)   D + D → T (1.01 MeV) + p (3.02 MeV)                     Q = 4.03 MeV
(DDn)   D + D → 3He (0.82 MeV) + n (2.45 MeV)                   Q = 3.27 MeV
(DHe3)  D + 3He → α (3.6 MeV) + p (14.7 MeV)                    Q = 18.3 MeV
(TT)    T + T → 2n + α                                          Q = 11.3 MeV
(He3He3)3He + 3He → 2p  + α                                     Q = 12.86 MeV
(He3Tpn)3He + T → α (4.8 MeV) + n (14.1 MeV)                    Q = 12.1 MeV
(He3TD) 3He + T → α (4.8 MeV) + D (9.5 MeV)                     Q = 14.3 MeV
(He3Dnp)3He + D → α (0.5 MeV) + n (1.9 MeV) + p (11.9 MeV)      Q = 14.3 MeV

other fusion reactions (p-B11,...) are not considered since the focus in on magnetically confined plasmas.
"""
from __future__ import annotations

from typing import Any

from fusdb import relation
from fusdb.registry import (
    DD_HE3_ENERGY_J,
    DD_N_ENERGY_J,
    DD_P_ENERGY_J,
    DD_T_ENERGY_J,
    DHE3_ALPHA_ENERGY_J,
    DHE3_P_ENERGY_J,
    DT_ALPHA_ENERGY_J,
    DT_N_ENERGY_J,
    HE3HE3_ALPHA_ENERGY_J,
    HE3HE3_P_ENERGY_J,
    THE3_D_ALPHA_ENERGY_J,
    THE3_D_D_ENERGY_J,
    THE3_NP_ALPHA_ENERGY_J,
    THE3_NP_N_ENERGY_J,
    THE3_NP_P_ENERGY_J,
    TT_ALPHA_ENERGY_J,
    TT_N_ENERGY_J,
)

@relation(
    name='Total fusion power',
    tags=('fusion_power',),
    outputs='P_fus',
)
def fusion_power_total(
    P_fus_DT: float,
    P_fus_DD: float,
    P_fus_DHe3: float,
    P_fus_TT: float,
    P_fus_He3He3: float = 0.0,
    P_fus_THe3: float = 0.0,
 ) -> Any:
    """Return total fusion power from all implemented reaction channels."""
    return P_fus_DT + P_fus_DD + P_fus_DHe3 + P_fus_He3He3 + P_fus_THe3 + P_fus_TT


########################################
@relation(
    name='Charged fusion power',
    tags=('fusion_power',),
    outputs='P_charged',
)
def charged_fusion_power(
    P_fus_DT_alpha: float,
    P_fus_DDn_He3: float,
    P_fus_DDp_T: float,
    P_fus_DDp_p: float,
    P_fus_DHe3_alpha: float,
    P_fus_DHe3_p: float,
    P_fus_He3He3_alpha: float = 0.0,
    P_fus_He3He3_p: float = 0.0,
    P_fus_THe3_D_alpha: float = 0.0,
    P_fus_THe3_D_D: float = 0.0,
    P_fus_THe3_np_alpha: float = 0.0,
    P_fus_THe3_np_p: float = 0.0,
    P_fus_TT_alpha: float = 0.0,
 ) -> Any:
    """Return charged-particle fusion power from all implemented channels."""
    return (
        P_fus_DT_alpha
        + P_fus_DDn_He3
        + P_fus_DDp_T
        + P_fus_DDp_p
        + P_fus_DHe3_alpha
        + P_fus_DHe3_p
        + P_fus_He3He3_alpha
        + P_fus_He3He3_p
        + P_fus_THe3_D_alpha
        + P_fus_THe3_D_D
        + P_fus_THe3_np_alpha
        + P_fus_THe3_np_p
        + P_fus_TT_alpha
    )
    
########################################
@relation(
    name='Neutron fusion power',
    tags=('fusion_power',),
    outputs='P_neutron',
)
def neutron_fusion_power(
    P_fus_DT_n: float,
    P_fus_DDn_n: float,
    P_fus_THe3_np_n: float = 0.0,
    P_fus_TT_n: float = 0.0,
 ) -> Any:
    """Return neutron fusion power from all implemented neutron-producing channels."""
    return P_fus_DT_n + P_fus_DDn_n + P_fus_THe3_np_n + P_fus_TT_n


################### DT #####################
########################################
@relation(
    name='DT fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DT',
)
def fusion_power_dt(P_fus_DT_alpha: float, P_fus_DT_n: float) -> Any:
    """Return total D-T fusion power from alpha and neutron components."""
    return P_fus_DT_alpha + P_fus_DT_n


########################################
@relation(
    name='DT alpha power',
    tags=('fusion_power',),
    outputs='P_fus_DT_alpha',
)
def alpha_power_dt(Rr_DT: float) -> Any:
    """Return alpha power from D-T fusion."""
    return DT_ALPHA_ENERGY_J * Rr_DT

########################################
@relation(
    name='DT neutron power',
    tags=('fusion_power',),
    outputs='P_fus_DT_n',
)
def neutron_power_dt(Rr_DT: float) -> Any:
    """Return neutron power from D-T fusion."""
    return DT_N_ENERGY_J * Rr_DT


#################### DD #####################
########################################
@relation(
    name='DD fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DD',
)
def fusion_power_dd(P_fus_DDn: float, P_fus_DDp: float) -> Any:
    """Return total D-D fusion power from both branches."""
    return P_fus_DDn + P_fus_DDp


########################################
@relation(
    name='DD (T+p) fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DDp',
)
def fusion_power_ddp(P_fus_DDp_T: float, P_fus_DDp_p: float) -> Any:
    """Return D-D fusion power from the T+p branch."""
    return P_fus_DDp_T + P_fus_DDp_p

########################################
@relation(
    name='DD (He3+n) fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DDn',
)
def fusion_power_ddn(P_fus_DDn_He3: float, P_fus_DDn_n: float) -> Any:
    """Return D-D fusion power from the He3+n branch."""
    return P_fus_DDn_He3 + P_fus_DDn_n


########################################
@relation(
    name='DDp triton power',
    tags=('fusion_power',),
    outputs='P_fus_DDp_T',
)
def triton_power_dd(Rr_DDp: float) -> Any:
    """Return triton power from the D(d,p)T branch."""
    return DD_T_ENERGY_J * Rr_DDp

########################################
@relation(
    name='DDp proton power',
    tags=('fusion_power',),
    outputs='P_fus_DDp_p',
)
def proton_power_dd(Rr_DDp: float) -> Any:
    """Return proton power from the D(d,p)T branch."""
    return DD_P_ENERGY_J * Rr_DDp

########################################
@relation(
    name='DDn helium-3 power',
    tags=('fusion_power',),
    outputs='P_fus_DDn_He3',
)
def he3_power_dd(Rr_DDn: float) -> Any:
    """Return He3 power from the D(d,n)He3 branch."""
    return DD_HE3_ENERGY_J * Rr_DDn

########################################
@relation(
    name='DDn neutron power',
    tags=('fusion_power',),
    outputs='P_fus_DDn_n',
)
def neutron_power_dd(Rr_DDn: float) -> Any:
    """Return neutron power from the D(d,n)He3 branch."""
    return DD_N_ENERGY_J * Rr_DDn

################ DHe3 ####################
########################################
@relation(
    name='D-He3 fusion power',
    tags=('fusion_power',),
    outputs='P_fus_DHe3',
)
def fusion_power_dhe3(P_fus_DHe3_alpha: float, P_fus_DHe3_p: float) -> Any:
    """Return total D-He3 fusion power from alpha and proton components."""
    return P_fus_DHe3_alpha + P_fus_DHe3_p


########################################
@relation(
    name='D-He3 alpha power',
    tags=('fusion_power',),
    outputs='P_fus_DHe3_alpha',
)
def alpha_power_dhe3(Rr_DHe3: float) -> Any:
    """Return alpha power from D-He3 fusion."""
    return DHE3_ALPHA_ENERGY_J * Rr_DHe3


########################################
@relation(
    name='D-He3 proton power',
    tags=('fusion_power',),
    outputs='P_fus_DHe3_p',
)
def proton_power_dhe3(Rr_DHe3: float) -> Any:
    """Return proton power from D-He3 fusion."""
    return DHE3_P_ENERGY_J * Rr_DHe3

################### He3He3 #####################
########################################
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


########################################
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


########################################
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

#################### TT #####################
########################################
@relation(
    name='TT fusion power',
    tags=('fusion_power',),
    outputs='P_fus_TT',
)
def fusion_power_tt(P_fus_TT_alpha: float, P_fus_TT_n: float) -> Any:
    """Return total T-T fusion power from alpha and neutron components."""
    return P_fus_TT_alpha + P_fus_TT_n


########################################
@relation(
    name='TT alpha power',
    tags=('fusion_power',),
    outputs='P_fus_TT_alpha',
)
def alpha_power_tt(Rr_TT: float) -> Any:
    """Return alpha power from T-T fusion."""
    return TT_ALPHA_ENERGY_J * Rr_TT


########################################
@relation(
    name='TT neutron power',
    tags=('fusion_power',),
    outputs='P_fus_TT_n',
)
def neutron_power_tt(Rr_TT: float) -> Any:
    """Return combined neutron power from T-T fusion."""
    return TT_N_ENERGY_J * Rr_TT


################### THe3 #####################
########################################
@relation(
    name='T-He3 fusion power',
    tags=('fusion_power',),
    outputs='P_fus_THe3',
)
def fusion_power_the3(P_fus_THe3_D: float, P_fus_THe3_np: float) -> Any:
    """Return total T-He3 fusion power from the D and n+p branches."""
    return P_fus_THe3_D + P_fus_THe3_np


########################################
@relation(
    name='T-He3 alpha+D fusion power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_D',
)
def fusion_power_the3_d(P_fus_THe3_D_alpha: float, P_fus_THe3_D_D: float) -> Any:
    """Return T-He3 alpha+D branch power from alpha and deuteron components."""
    return P_fus_THe3_D_alpha + P_fus_THe3_D_D


########################################
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


########################################
@relation(
    name='T-He3 alpha+D alpha power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_D_alpha',
)
def alpha_power_the3_d(Rr_THe3_D: float) -> Any:
    """Return alpha power from the T-He3 alpha+D branch."""
    return THE3_D_ALPHA_ENERGY_J * Rr_THe3_D


########################################
@relation(
    name='T-He3 alpha+D deuteron power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_D_D',
)
def deuteron_power_the3_d(Rr_THe3_D: float) -> Any:
    """Return deuteron power from the T-He3 alpha+D branch."""
    return THE3_D_D_ENERGY_J * Rr_THe3_D


########################################
@relation(
    name='T-He3 alpha+n+p alpha power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np_alpha',
)
def alpha_power_the3_np(Rr_THe3_np: float) -> Any:
    """Return alpha power from the T-He3 alpha+n+p branch."""
    return THE3_NP_ALPHA_ENERGY_J * Rr_THe3_np


########################################
@relation(
    name='T-He3 alpha+n+p neutron power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np_n',
)
def neutron_power_the3_np(Rr_THe3_np: float) -> Any:
    """Return neutron power from the T-He3 alpha+n+p branch."""
    return THE3_NP_N_ENERGY_J * Rr_THe3_np


########################################
@relation(
    name='T-He3 alpha+n+p proton power',
    tags=('fusion_power',),
    outputs='P_fus_THe3_np_p',
)
def proton_power_the3_np(Rr_THe3_np: float) -> Any:
    """Return proton power from the T-He3 alpha+n+p branch."""
    return THE3_NP_P_ENERGY_J * Rr_THe3_np
