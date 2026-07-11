"""DT reaction-rate relations."""

from typing import Any

from fusdb.utils import trapezoid

from fusdb.relation import relation


@relation(
    name='DT reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DT',
)
def reaction_rate_dt(n_D: float, n_T: float, sigmav_DT: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DT reaction rate.

    Args:
        n_D: Deuterium density profile.
        n_T: Tritium density profile.
        sigmav_DT: DT reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DT reaction rate in 1/s.
    """
    # Form the local DT reaction-rate density.
    integrand = n_D * n_T * sigmav_DT

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
