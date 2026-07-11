"""DD reaction-rate relations."""

from typing import Any

from fusdb.utils import trapezoid

from fusdb.relation import relation


@relation(
    name='DD (He3+n) reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DDn',
)
def reaction_rate_ddn(n_D: float, sigmav_DDn: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DDn reaction rate.

    Args:
        n_D: Deuterium density profile.
        sigmav_DDn: DDn reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DDn reaction rate in 1/s.
    """
    # Form the local DDn reaction-rate density.
    integrand = 0.5 * (n_D**2) * sigmav_DDn

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)


@relation(
    name='DD (T+p) reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DDp',
)
def reaction_rate_ddp(n_D: float, sigmav_DDp: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DDp reaction rate.

    Args:
        n_D: Deuterium density profile.
        sigmav_DDp: DDp reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DDp reaction rate in 1/s.
    """
    # Form the local DDp reaction-rate density.
    integrand = 0.5 * (n_D**2) * sigmav_DDp

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
