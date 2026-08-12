"""D-He3 reaction-rate relations."""

from typing import Any

from fusdb.utils import volume_average

from fusdb.relation import relation


@relation(
    name='D-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DHe3',
)
def reaction_rate_dhe3(
    n_D: float,
    n_He3: float,
    sigmav_DHe3: float,
    V_p: float,
    rho: float,
    w_V: Any = None,
) -> Any:
    """Return the volume-integrated D-He3 reaction rate."""
    integrand = n_D * n_He3 * sigmav_DHe3
    return V_p * volume_average(integrand, rho, weight=w_V)
