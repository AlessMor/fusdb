"""DT reaction-rate relations."""

from typing import Any

from fusdb.utils import volume_average

from fusdb.relation import relation


@relation(
    name='DT reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DT',
)
def reaction_rate_dt(
    n_D: float,
    n_T: float,
    sigmav_DT: float,
    V_p: float,
    rho: float,
    w_V: Any = None,
) -> Any:
    """Return the volume-integrated DT reaction rate.

    ``rho`` is the common computational grid. ``w_V`` is the optional physical
    volume measure; omitting it retains the historical self-similar weighting.
    """
    integrand = n_D * n_T * sigmav_DT
    return V_p * volume_average(integrand, rho, weight=w_V)
