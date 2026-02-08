"""
The most favourable fusion reactions (up to 3He):
(DT)    D + T → α (3.52 MeV) + n (14.08 MeV)                    Q = 17.6 MeV
(DDp)   D + D → T (1.01 MeV) + p (3.02 MeV)                     Q = 4.03 MeV
(DDn)   D + D → 3He (0.82 MeV) + n (2.45 MeV)                   Q = 3.27 MeV
(DHe3)  D + 3He → α (3.6 MeV) + p (14.7 MeV)                    Q = 18.3 MeV
(TT)    T + T → 2n + α                                          Q = 11.3 MeV
(He3He3)3He + 3He → 2p  + α                
(He3Tpn)3He + T → α (4.8 MeV) + n (14.1 MeV)                    Q = 12.1 MeV
(He3TD) 3He + T → α (4.8 MeV) + D (9.5 MeV)                     Q = 14.3 MeV
(He3Dnp)3He + D → α (0.5 MeV) + n (1.9 MeV) + p (11.9 MeV)      Q = 14.3 MeV

other fusion reactions (p-B11,...) are not considered since the focus in on magnetically confined plasmas.
"""
# NOTE: DT, DD, DHe3 are implemented. TT, He3He3, THe3 to be implemented and energies for each product needs to be defined.
# cite references for energies

from __future__ import annotations

from fusdb.relation_class import Relation_decorator as Relation
from fusdb.registry import (
    DD_HE3_ENERGY_J,
    DD_N_ENERGY_J,
    DD_P_ENERGY_J,
    DD_T_ENERGY_J,
    DHE3_ALPHA_ENERGY_J,
    DHE3_P_ENERGY_J,
    DT_ALPHA_ENERGY_J,
    DT_N_ENERGY_J,
    TT_REACTION_ENERGY_J,
)
from fusdb.relations.power_balance.fusion_power.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3,
    sigmav_THe3,
    sigmav_TT,
)

@Relation(
    name="Total fusion power",
    output="P_fus",
    tags=("fusion_power",),
    constraints=("P_fus >= 0",),
)
def fusion_power_total(
    P_fus_DT: float,
    P_fus_DD: float,
    P_fus_DHe3: float,
    P_fus_TT: float,
) -> float:
    """Return total fusion power from DT, DD, D-He3, and TT contributions."""
    return P_fus_DT + P_fus_DD + P_fus_DHe3 + P_fus_TT
# TODO(low): Implement total fusion power from He3He3, He3Tpn, He3TD, He3Dnp


########################################
@Relation(
    name="Charged fusion power",
    output="P_charged",
    tags=("fusion_power",),
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
# TODO(low): Implement charged fusion power from TT, He3He3, He3Tpn, He3TD, He3Tnp
    
########################################
@Relation(
    name="Neutron fusion power",
    output="P_neutron",
    tags=("fusion_power",),
)
def neutron_fusion_power(
    P_fus_DT_n: float,
    P_fus_DDn_n: float,
) -> float:
    """Return neutron fusion power from common D-T and D-D channels."""
    return P_fus_DT_n + P_fus_DDn_n
#TODO(low): Implement neutron fusion power from TT, He3Tpn, He3Tnp


################### DT #####################
########################################
@Relation(
    name="DT fusion power",
    output="P_fus_DT",
    tags=("fusion_power",),
)
def fusion_power_dt(P_fus_DT_alpha: float, P_fus_DT_n: float) -> float:
    """Return total D-T fusion power from alpha and neutron components."""
    return P_fus_DT_alpha + P_fus_DT_n


########################################
@Relation(
    name="DT alpha power",
    output="P_fus_DT_alpha",
    tags=("fusion_power",),
)
def alpha_power_dt(Rr_DT: float) -> float:
    """Return alpha power from D-T fusion."""
    return DT_ALPHA_ENERGY_J * Rr_DT

########################################
@Relation(
    name="DT neutron power",
    output="P_fus_DT_n",
    tags=("fusion_power",),
)
def neutron_power_dt(Rr_DT: float) -> float:
    """Return neutron power from D-T fusion."""
    return DT_N_ENERGY_J * Rr_DT


#################### DD #####################
########################################
@Relation(
    name="DD fusion power",
    output="P_fus_DD",
    tags=("fusion_power",),
)
def fusion_power_dd(P_fus_DDn: float, P_fus_DDp: float) -> float:
    """Return total D-D fusion power from both branches."""
    return P_fus_DDn + P_fus_DDp


########################################
@Relation(
    name="DD (T+p) fusion power",
    output="P_fus_DDp",
    tags=("fusion_power",),
)
def fusion_power_ddp(P_fus_DDp_T: float, P_fus_DDp_p: float) -> float:
    """Return D-D fusion power from the T+p branch."""
    return P_fus_DDp_T + P_fus_DDp_p

########################################
@Relation(
    name="DD (He3+n) fusion power",
    output="P_fus_DDn",
    tags=("fusion_power",),
)
def fusion_power_ddn(P_fus_DDn_He3: float, P_fus_DDn_n: float) -> float:
    """Return D-D fusion power from the He3+n branch."""
    return P_fus_DDn_He3 + P_fus_DDn_n


########################################
@Relation(
    name="DDp triton power",
    output="P_fus_DDp_T",
    tags=("fusion_power",),
)
def triton_power_dd(Rr_DDp: float) -> float:
    """Return triton power from the D(d,p)T branch."""
    return DD_T_ENERGY_J * Rr_DDp

########################################
@Relation(
    name="DDp proton power",
    output="P_fus_DDp_p",
    tags=("fusion_power",),
)
def proton_power_dd(Rr_DDp: float) -> float:
    """Return proton power from the D(d,p)T branch."""
    return DD_P_ENERGY_J * Rr_DDp

########################################
@Relation(
    name="DDn helium-3 power",
    output="P_fus_DDn_He3",
    tags=("fusion_power",),
)
def he3_power_dd(Rr_DDn: float) -> float:
    """Return He3 power from the D(d,n)He3 branch."""
    return DD_HE3_ENERGY_J * Rr_DDn

########################################
@Relation(
    name="DDn neutron power",
    output="P_fus_DDn_n",
    tags=("fusion_power",),
)
def neutron_power_dd(Rr_DDn: float) -> float:
    """Return neutron power from the D(d,n)He3 branch."""
    return DD_N_ENERGY_J * Rr_DDn

################ DHe3 ####################
########################################
@Relation(
    name="D-He3 fusion power",
    output="P_fus_DHe3",
    tags=("fusion_power",),
)
def fusion_power_dhe3(P_fus_DHe3_alpha: float, P_fus_DHe3_p: float) -> float:
    """Return total D-He3 fusion power from alpha and proton components."""
    return P_fus_DHe3_alpha + P_fus_DHe3_p


########################################
@Relation(
    name="D-He3 alpha power",
    output="P_fus_DHe3_alpha",
    tags=("fusion_power",),
)
def alpha_power_dhe3(Rr_DHe3: float) -> float:
    """Return alpha power from D-He3 fusion."""
    return DHE3_ALPHA_ENERGY_J * Rr_DHe3


########################################
@Relation(
    name="D-He3 proton power",
    output="P_fus_DHe3_p",
    tags=("fusion_power",),
)
def proton_power_dhe3(Rr_DHe3: float) -> float:
    """Return proton power from D-He3 fusion."""
    return DHE3_P_ENERGY_J * Rr_DHe3

################### He3He3 #####################
#TODO(low): Implement He3He3 relations

#################### TT #####################
#TODO(low): Implement TT relations
########################################
@Relation(
    name="TT fusion power",
    output="P_fus_TT",
    tags=("fusion_power",),
)
def fusion_power_tt(Rr_TT: float) -> float:
    """Return total fusion power from T-T reactions."""
    return TT_REACTION_ENERGY_J * Rr_TT

