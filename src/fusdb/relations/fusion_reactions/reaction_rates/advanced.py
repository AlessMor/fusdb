"""Advanced-fuel reaction-rate relations."""

from typing import Any

import numpy as np

from fusdb.numerics import volume_average

from fusdb.relation import relation


@relation(
    name='He3-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_He3He3',
)
def reaction_rate_he3he3(
    n_He3: float, sigmav_He3He3: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated He3He3 reaction rate.

    Args:
        n_He3: Helium-3 density profile.
        sigmav_He3He3: He3He3 reactivity profile.
        V_p: Plasma volume.
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Returns:
        The total He3He3 reaction rate in 1/s.
    """
    # Form the local He3He3 reaction-rate density.
    integrand = 0.5 * (n_He3**2) * sigmav_He3He3

    # Integrate the profile over the plasma volume.
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='T-He3 alpha+D reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3_D',
)
def reaction_rate_the3_d(
    n_T: float, n_He3: float, sigmav_THe3_D: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated THe3-to-D reaction rate.

    Args:
        n_T: Tritium density profile.
        n_He3: Helium-3 density profile.
        sigmav_THe3_D: THe3_D reactivity profile.
        V_p: Plasma volume.
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Returns:
        The total THe3_D reaction rate in 1/s.
    """
    # Form the local THe3_D reaction-rate density.
    integrand = n_T * n_He3 * sigmav_THe3_D

    # Integrate the profile over the plasma volume.
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='T-He3 alpha+n+p reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3_np',
)
def reaction_rate_the3_np(
    n_T: float, n_He3: float, sigmav_THe3_np: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated THe3-to-np reaction rate.

    Args:
        n_T: Tritium density profile.
        n_He3: Helium-3 density profile.
        sigmav_THe3_np: THe3_np reactivity profile.
        V_p: Plasma volume.
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Returns:
        The total THe3_np reaction rate in 1/s.
    """
    # Form the local THe3_np reaction-rate density.
    integrand = n_T * n_He3 * sigmav_THe3_np

    # Integrate the profile over the plasma volume.
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(
    name='T-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3',
)
def reaction_rate_the3(Rr_THe3_D: float, Rr_THe3_np: float) -> Any:
    """Return the total THe3 reaction rate from the implemented branches.

    Args:
        Rr_THe3_D: THe3-to-D branch reaction rate.
        Rr_THe3_np: THe3-to-np branch reaction rate.

    Returns:
        The total THe3 reaction rate in 1/s.
    """
    # Sum the implemented THe3 branch rates.
    return Rr_THe3_D + Rr_THe3_np


@relation(
    name='T-T reaction rate',
    tags=('fusion_power',),
    outputs='Rr_TT',
)
def reaction_rate_tt(
    n_T: float, sigmav_TT: float, V_p: float, rho: float, w_V: Any = None
) -> Any:
    """Return the volume-integrated TT reaction rate.

    Args:
        n_T: Tritium density profile.
        sigmav_TT: TT reactivity profile.
        V_p: Plasma volume.
        rho: Common computational profile grid.
        w_V: Optional physical volume-integration weight on ``rho``.

    Returns:
        The total TT reaction rate in 1/s.
    """
    # Form the local TT reaction-rate density.
    integrand = 0.5 * (n_T**2) * sigmav_TT

    # Integrate the profile over the plasma volume.
    return V_p * volume_average(integrand, rho, weight=w_V)


@relation(name="p-B11 reaction rate", tags=("fusion_power",), outputs="Rr_pB11")
def reaction_rate_pb11(
    n_p: Any,
    n_B11: Any,
    sigmav_pB11: Any,
    V_p: Any,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Volume-integrated p + B11 reaction rate.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    local = np.asarray(n_p) * np.asarray(n_B11) * np.asarray(sigmav_pB11)
    return np.asarray(V_p) * volume_average(local, rho, weight=w_V)
