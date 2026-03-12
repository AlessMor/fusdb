"""Profile reactivity relations used by explicit fusion-rate pipelines."""

from __future__ import annotations

from fusdb.relation_util import relation
from fusdb.relations.power_balance.fusion_power.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3,
    sigmav_THe3,
    sigmav_TT,
)


@relation(
    name="DT reactivity profile",
    output="sigmav_DT_profile",
    tags=("fusion_power",),
)
def sigmav_dt_profile(T_i: float) -> float:
    """Return DT reactivity profile [m^3/s] from ion temperature profile."""
    return sigmav_DT_BoschHale(T_i)


@relation(
    name="DDn reactivity profile",
    output="sigmav_DDn_profile",
    tags=("fusion_power",),
)
def sigmav_ddn_profile(T_i: float) -> float:
    """Return DD (He3+n) branch reactivity profile [m^3/s]."""
    _, sigmav_ddn, _ = sigmav_DD_BoschHale(T_i)
    return sigmav_ddn


@relation(
    name="DDp reactivity profile",
    output="sigmav_DDp_profile",
    tags=("fusion_power",),
)
def sigmav_ddp_profile(T_i: float) -> float:
    """Return DD (T+p) branch reactivity profile [m^3/s]."""
    _, _, sigmav_ddp = sigmav_DD_BoschHale(T_i)
    return sigmav_ddp


@relation(
    name="DHe3 reactivity profile",
    output="sigmav_DHe3_profile",
    tags=("fusion_power",),
)
def sigmav_dhe3_profile(T_i: float) -> float:
    """Return D-He3 reactivity profile [m^3/s]."""
    return sigmav_DHe3_BoschHale(T_i)


@relation(
    name="TT reactivity profile",
    output="sigmav_TT_profile",
    tags=("fusion_power",),
)
def sigmav_tt_profile(T_i: float) -> float:
    """Return TT reactivity profile [m^3/s]."""
    return sigmav_TT(T_i)


@relation(
    name="He3He3 reactivity profile",
    output="sigmav_He3He3_profile",
    tags=("fusion_power",),
)
def sigmav_he3he3_profile(T_i: float) -> float:
    """Return He3-He3 reactivity profile [m^3/s]."""
    return sigmav_He3He3(T_i)


@relation(
    name="THe3 reactivity profile",
    output="sigmav_THe3_profile",
    tags=("fusion_power",),
)
def sigmav_the3_profile(T_i: float) -> float:
    """Return T-He3 reactivity profile [m^3/s]."""
    return sigmav_THe3(T_i)
