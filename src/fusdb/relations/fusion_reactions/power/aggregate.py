"""Aggregate fusion-power relations."""

from typing import Any

from fusdb.relation import relation


@relation(
    name='Total fusion power',
    tags=('fusion_power',),
    outputs='P_fus',
)
def fusion_power_total(
    P_fus_DT: float,
    P_fus_DD: float,
    P_fus_TT: float,
    P_fus_DHe3: float = 0.0,
    P_fus_He3He3: float = 0.0,
    P_fus_THe3: float = 0.0,
 ) -> Any:
    """Return total fusion power from all implemented reaction channels.

    Only the always-present D/T base channels (DT, DD, TT) are required; the
    He3-bearing channels are optional and contribute zero when their species
    are absent and the channel has been pruned from the system.
    """
    return P_fus_DT + P_fus_DD + P_fus_DHe3 + P_fus_He3He3 + P_fus_THe3 + P_fus_TT


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
    P_fus_DHe3_alpha: float = 0.0,
    P_fus_DHe3_p: float = 0.0,
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
