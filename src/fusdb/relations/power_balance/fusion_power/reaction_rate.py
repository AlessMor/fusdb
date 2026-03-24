"""Fusion reaction-rate relations using explicit profile integration."""

from __future__ import annotations

from fusdb.relation_util import relation
from fusdb.utils import integrate_profile


################## TOTAL REACTION RATES ##################


@relation(
    name="DT reaction rate",
    output="Rr_DT",
    tags=("fusion_power",),
)
def reaction_rate_dt(n_D: float, n_T: float, sigmav_DT: float, V_p: float) -> float:
    """Return volume-integrated DT reaction rate [1/s]."""
    integrand = n_D * n_T * sigmav_DT
    return integrate_profile(integrand, V_p, error_label="DT reaction-rate")


@relation(
    name="DD (He3+n) reaction rate",
    output="Rr_DDn",
    tags=("fusion_power",),
)
def reaction_rate_ddn(n_D: float, sigmav_DDn: float, V_p: float) -> float:
    """Return volume-integrated D(d,n)He3 reaction rate [1/s]."""
    integrand = 0.5 * (n_D ** 2) * sigmav_DDn
    return integrate_profile(integrand, V_p, error_label="DDn reaction-rate")


@relation(
    name="DD (T+p) reaction rate",
    output="Rr_DDp",
    tags=("fusion_power",),
)
def reaction_rate_ddp(n_D: float, sigmav_DDp: float, V_p: float) -> float:
    """Return volume-integrated D(d,p)T reaction rate [1/s]."""
    integrand = 0.5 * (n_D ** 2) * sigmav_DDp
    return integrate_profile(integrand, V_p, error_label="DDp reaction-rate")


@relation(
    name="D-He3 reaction rate",
    output="Rr_DHe3",
    tags=("fusion_power",),
)
def reaction_rate_dhe3(n_D: float, n_He3: float, sigmav_DHe3: float, V_p: float) -> float:
    """Return volume-integrated D-He3 reaction rate [1/s]."""
    integrand = n_D * n_He3 * sigmav_DHe3
    return integrate_profile(integrand, V_p, error_label="DHe3 reaction-rate")


@relation(
    name="T-T reaction rate",
    output="Rr_TT",
    tags=("fusion_power",),
)
def reaction_rate_tt(n_T: float, sigmav_TT: float, V_p: float) -> float:
    """Return volume-integrated T-T reaction rate [1/s]."""
    integrand = 0.5 * (n_T ** 2) * sigmav_TT
    return integrate_profile(integrand, V_p, error_label="TT reaction-rate")


@relation(
    name="He3-He3 reaction rate",
    output="Rr_He3He3",
    tags=("fusion_power",),
)
def reaction_rate_he3he3(n_He3: float, sigmav_He3He3: float, V_p: float) -> float:
    """Return volume-integrated He3-He3 reaction rate [1/s]."""
    integrand = 0.5 * (n_He3 ** 2) * sigmav_He3He3
    return integrate_profile(integrand, V_p, error_label="He3He3 reaction-rate")


@relation(
    name="T-He3 alpha+D reaction rate",
    output="Rr_THe3_D",
    tags=("fusion_power",),
)
def reaction_rate_the3_d(n_T: float, n_He3: float, sigmav_THe3_D: float, V_p: float) -> float:
    """Return volume-integrated T-He3 alpha + D branch reaction rate [1/s]."""
    integrand = n_T * n_He3 * sigmav_THe3_D
    return integrate_profile(integrand, V_p, error_label="THe3_D reaction-rate")


@relation(
    name="T-He3 alpha+n+p reaction rate",
    output="Rr_THe3_np",
    tags=("fusion_power",),
)
def reaction_rate_the3_np(n_T: float, n_He3: float, sigmav_THe3_np: float, V_p: float) -> float:
    """Return volume-integrated T-He3 alpha + n + p branch reaction rate [1/s]."""
    integrand = n_T * n_He3 * sigmav_THe3_np
    return integrate_profile(integrand, V_p, error_label="THe3_np reaction-rate")


@relation(
    name="T-He3 reaction rate",
    output="Rr_THe3",
    tags=("fusion_power",),
)
def reaction_rate_the3(Rr_THe3_D: float, Rr_THe3_np: float) -> float:
    """Return total T-He3 reaction rate from the two implemented branches."""
    return Rr_THe3_D + Rr_THe3_np
