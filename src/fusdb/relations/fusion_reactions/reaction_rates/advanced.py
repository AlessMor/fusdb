"""Advanced-fuel reaction-rate relations."""

from typing import Any

from fusdb.utils import volume_average

from fusdb.relation import relation


@relation(
    name='He3-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_He3He3',
)
def reaction_rate_he3he3(
    n_He3: float, sigmav_He3He3: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated He3-He3 reaction rate."""
    integrand = 0.5 * (n_He3**2) * sigmav_He3He3
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='T-He3 alpha+D reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3_D',
)
def reaction_rate_the3_d(
    n_T: float,
    n_He3: float,
    sigmav_THe3_D: float,
    V_p: float,
    rho: float,
    w_V: Any = None,
) -> Any:
    """Return the volume-integrated T-He3-to-D reaction rate."""
    integrand = n_T * n_He3 * sigmav_THe3_D
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='T-He3 alpha+n+p reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3_np',
)
def reaction_rate_the3_np(
    n_T: float,
    n_He3: float,
    sigmav_THe3_np: float,
    V_p: float,
    rho: float,
    w_V: Any = None,
) -> Any:
    """Return the volume-integrated T-He3-to-np reaction rate."""
    integrand = n_T * n_He3 * sigmav_THe3_np
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='T-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3',
)
def reaction_rate_the3(Rr_THe3_D: float, Rr_THe3_np: float) -> Any:
    """Return the total T-He3 reaction rate from the implemented branches."""
    return Rr_THe3_D + Rr_THe3_np


@relation(
    name='T-T reaction rate',
    tags=('fusion_power',),
    outputs='Rr_TT',
)
def reaction_rate_tt(
    n_T: float, sigmav_TT: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated T-T reaction rate."""
    integrand = 0.5 * (n_T**2) * sigmav_TT
    return V_p * volume_average(integrand, rho, weight=w_V)
