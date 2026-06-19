"""D-He3 reaction-rate relations."""

from typing import Any

from scipy.integrate import trapezoid

from fusdb import relation


@relation(
    name='D-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DHe3',
)
def reaction_rate_dhe3(n_D: float, n_He3: float, sigmav_DHe3: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DHe3 reaction rate.

    Args:
        n_D: Deuterium density profile.
        n_He3: Helium-3 density profile.
        sigmav_DHe3: DHe3 reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DHe3 reaction rate in 1/s.
    """
    # Form the local DHe3 reaction-rate density.
    integrand = n_D * n_He3 * sigmav_DHe3

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
