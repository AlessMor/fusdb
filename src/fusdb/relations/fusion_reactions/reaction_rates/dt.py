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
    n_D: float, n_T: float, sigmav_DT: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated DT reaction rate.

    Args:
        n_D: Deuterium density profile.
        n_T: Tritium density profile.
        sigmav_DT: DT reactivity profile.
        V_p: Plasma volume.
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Returns:
        The total DT reaction rate in 1/s.
    """
    # Form the local DT reaction-rate density.
    integrand = n_D * n_T * sigmav_DT

    # Integrate the profile over the plasma volume.
    return V_p * volume_average(integrand, rho, weight=w_V)
