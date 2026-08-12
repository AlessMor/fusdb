"""DD reaction-rate relations."""

from typing import Any

from fusdb.utils import volume_average

from fusdb.relation import relation


@relation(
    name='DD (He3+n) reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DDn',
)
def reaction_rate_ddn(
    n_D: float, sigmav_DDn: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated DDn reaction rate."""
    integrand = 0.5 * (n_D**2) * sigmav_DDn
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='DD (T+p) reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DDp',
)
def reaction_rate_ddp(
    n_D: float, sigmav_DDp: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated DDp reaction rate."""
    integrand = 0.5 * (n_D**2) * sigmav_DDp
    return V_p * volume_average(integrand, rho, weight=w_V)
