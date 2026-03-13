"""Profile reactivity relations used by explicit fusion-rate pipelines."""

from __future__ import annotations

from fusdb.relation_util import relation
from fusdb.relations.reactivities.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DD_ENDFB_VIII0,
    sigmav_DHe3_BoschHale,
    sigmav_DHe3_ENDFB_VIII0,
    sigmav_DHe3_NRL,
    sigmav_DT_BoschHale,
    sigmav_DT_ENDFB_VIII0,
    sigmav_DT_NRL,
    sigmav_He3He3,
    sigmav_He3He3_CF88,
    sigmav_He3He3_ENDFB_VIII0,
    sigmav_THe3,
    sigmav_THe3_CF88,
    sigmav_THe3_ENDFB_VIII0,
    sigmav_THe3_NRL,
    sigmav_TT,
    sigmav_TT_CF88,
    sigmav_TT_ENDFB_VIII0,
    sigmav_TT_NRL,
)


@relation(
    name="DT reactivity profile BoschHale",
    output="sigmav_DT_profile",
    tags=("fusion_power",),
)
def sigmav_dt_profile(T_i: float) -> float:
    """Return DT reactivity profile [m^3/s] from ion temperature profile."""
    return sigmav_DT_BoschHale(T_i)


@relation(
    name="DT reactivity profile ENDFB-VIII0",
    output="sigmav_DT_profile",
    tags=("fusion_power",),
)
def sigmav_dt_profile_endfb_viii0(T_i: float) -> float:
    """Return DT reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    return sigmav_DT_ENDFB_VIII0(T_i)


@relation(
    name="DT reactivity profile NRL",
    output="sigmav_DT_profile",
    tags=("fusion_power",),
)
def sigmav_dt_profile_nrl(T_i: float) -> float:
    """Return DT reactivity profile [m^3/s] using NRL Plasma Formulary tabulated data."""
    return sigmav_DT_NRL(T_i)


@relation(
    name="DDn reactivity profile BoschHale",
    output="sigmav_DDn_profile",
    tags=("fusion_power",),
)
def sigmav_ddn_profile(T_i: float) -> float:
    """Return DD (He3+n) branch reactivity profile [m^3/s]."""
    _, sigmav_ddn, _ = sigmav_DD_BoschHale(T_i)
    return sigmav_ddn


@relation(
    name="DDn reactivity profile ENDFB-VIII0",
    output="sigmav_DDn_profile",
    tags=("fusion_power",),
)
def sigmav_ddn_profile_endfb_viii0(T_i: float) -> float:
    """Return DD (He3+n) branch reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    _, sigmav_ddn, _ = sigmav_DD_ENDFB_VIII0(T_i)
    return sigmav_ddn


@relation(
    name="DDp reactivity profile BoschHale",
    output="sigmav_DDp_profile",
    tags=("fusion_power",),
)
def sigmav_ddp_profile(T_i: float) -> float:
    """Return DD (T+p) branch reactivity profile [m^3/s]."""
    _, _, sigmav_ddp = sigmav_DD_BoschHale(T_i)
    return sigmav_ddp


@relation(
    name="DDp reactivity profile ENDFB-VIII0",
    output="sigmav_DDp_profile",
    tags=("fusion_power",),
)
def sigmav_ddp_profile_endfb_viii0(T_i: float) -> float:
    """Return DD (T+p) branch reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    _, _, sigmav_ddp = sigmav_DD_ENDFB_VIII0(T_i)
    return sigmav_ddp


@relation(
    name="DHe3 reactivity profile BoschHale",
    output="sigmav_DHe3_profile",
    tags=("fusion_power",),
)
def sigmav_dhe3_profile(T_i: float) -> float:
    """Return D-He3 reactivity profile [m^3/s]."""
    return sigmav_DHe3_BoschHale(T_i)


@relation(
    name="DHe3 reactivity profile ENDFB-VIII0",
    output="sigmav_DHe3_profile",
    tags=("fusion_power",),
)
def sigmav_dhe3_profile_endfb_viii0(T_i: float) -> float:
    """Return D-He3 reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    return sigmav_DHe3_ENDFB_VIII0(T_i)


@relation(
    name="DHe3 reactivity profile NRL",
    output="sigmav_DHe3_profile",
    tags=("fusion_power",),
)
def sigmav_dhe3_profile_nrl(T_i: float) -> float:
    """Return D-He3 reactivity profile [m^3/s] using NRL Plasma Formulary tabulated data."""
    return sigmav_DHe3_NRL(T_i)


@relation(
    name="TT reactivity profile",
    output="sigmav_TT_profile",
    tags=("fusion_power",),
)
def sigmav_tt_profile(T_i: float) -> float:
    """Return TT reactivity profile [m^3/s]."""
    return sigmav_TT(T_i)


@relation(
    name="TT reactivity profile CF88",
    output="sigmav_TT_profile",
    tags=("fusion_power",),
)
def sigmav_tt_profile_cf88(T_i: float) -> float:
    """Return TT reactivity profile [m^3/s] using the CF88 parametrization."""
    return sigmav_TT_CF88(T_i)


@relation(
    name="TT reactivity profile ENDFB-VIII0",
    output="sigmav_TT_profile",
    tags=("fusion_power",),
)
def sigmav_tt_profile_endfb_viii0(T_i: float) -> float:
    """Return TT reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    return sigmav_TT_ENDFB_VIII0(T_i)


@relation(
    name="TT reactivity profile NRL",
    output="sigmav_TT_profile",
    tags=("fusion_power",),
)
def sigmav_tt_profile_nrl(T_i: float) -> float:
    """Return TT reactivity profile [m^3/s] using NRL Plasma Formulary tabulated data."""
    return sigmav_TT_NRL(T_i)


@relation(
    name="He3He3 reactivity profile",
    output="sigmav_He3He3_profile",
    tags=("fusion_power",),
)
def sigmav_he3he3_profile(T_i: float) -> float:
    """Return He3-He3 reactivity profile [m^3/s]."""
    return sigmav_He3He3(T_i)


@relation(
    name="He3He3 reactivity profile CF88",
    output="sigmav_He3He3_profile",
    tags=("fusion_power",),
)
def sigmav_he3he3_profile_cf88(T_i: float) -> float:
    """Return He3-He3 reactivity profile [m^3/s] using the CF88 parametrization."""
    return sigmav_He3He3_CF88(T_i)


@relation(
    name="He3He3 reactivity profile ENDFB-VIII0",
    output="sigmav_He3He3_profile",
    tags=("fusion_power",),
)
def sigmav_he3he3_profile_endfb_viii0(T_i: float) -> float:
    """Return He3-He3 reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    return sigmav_He3He3_ENDFB_VIII0(T_i)


@relation(
    name="THe3 reactivity profile",
    output="sigmav_THe3_profile",
    tags=("fusion_power",),
)
def sigmav_the3_profile(T_i: float) -> float:
    """Return T-He3 reactivity profile [m^3/s]."""
    return sigmav_THe3(T_i)


@relation(
    name="THe3 reactivity profile CF88",
    output="sigmav_THe3_profile",
    tags=("fusion_power",),
)
def sigmav_the3_profile_cf88(T_i: float) -> float:
    """Return T-He3 reactivity profile [m^3/s] using the CF88 parametrization."""
    sigmav_np, sigmav_d, sigmav_he5p = sigmav_THe3_CF88(T_i)
    return sigmav_np + sigmav_d + sigmav_he5p


@relation(
    name="THe3 reactivity profile ENDFB-VIII0",
    output="sigmav_THe3_profile",
    tags=("fusion_power",),
)
def sigmav_the3_profile_endfb_viii0(T_i: float) -> float:
    """Return T-He3 reactivity profile [m^3/s] using ENDF/B-VIII.0 tabulated data."""
    return sigmav_THe3_ENDFB_VIII0(T_i)


@relation(
    name="THe3 reactivity profile NRL",
    output="sigmav_THe3_profile",
    tags=("fusion_power",),
)
def sigmav_the3_profile_nrl(T_i: float) -> float:
    """Return T-He3 reactivity profile [m^3/s] using NRL Plasma Formulary tabulated data."""
    return sigmav_THe3_NRL(T_i)
