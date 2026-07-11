"""Sauter plasma geometry relations."""

import math

import numpy as np

from fusdb.relation import relation


@relation(
    name='Tokamak plasma poloidal length',
    tags=('geometry', 'tokamak'),
    outputs='L_p',
)
def sauter_plasma_cross_sectional_length(
    a: float,
    kappa: float,
    delta: float,
    squareness: float,
) -> float:
    """
    Poloidal length around the plasma cross-section.
    See :cite:`sauter_geometric_2016`.
    """
    # theta_07 reported in Sauter 2016 (:cite:`sauter_geometric_2016`) has been multiplied and divided by 1+sqrt(1+8*xi) to avoid division by 0
    theta_07 = -2 * squareness / (1 + np.sqrt(1 + 8 * squareness**2))
    # using the analytical formula for w_07, related to the radial width of the plasma shape at 70% of the maximum height
    w_07 = np.cos(theta_07 - squareness * np.sin(2 * theta_07)) / np.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    L_p = 2 * math.pi * a * (1 + 0.55 * (kappa - 1)) * (1 + 0.08 * delta**2) * (1 + 0.2 * (w_07 - 1))
    return  L_p


@relation(
    name='Tokamak plasma surface',
    tags=('geometry', 'tokamak'),
    outputs='A_p',
)
def sauter_plasma_surface(
    R: float,
    delta: float,
    eps: float,
    L_p: float,
) -> float:
    """
    Surface area around the LCFS in the toroidal and poloidal direction
    See :cite:`sauter_geometric_2016`.
    """
    A_p = 2 * math.pi * R * (1 - 0.32 * delta * eps) * L_p
    return A_p


@relation(
    name='Tokamak plasma volume',
    tags=('geometry', 'tokamak'),
    outputs='V_p',
)
def sauter_plasma_volume(
    R: float,
    delta: float,
    eps: float,
    S_phi: float,
) -> float:
    """
    Plasma Volume inside the LCFS
    See :cite:`sauter_geometric_2016`.
    """
    V_p = 2 * math.pi * R * (1 - 0.25 * delta * eps) * S_phi
    return V_p


@relation(
    name='Tokamak plasma cross-sectional surface',
    tags=('geometry', 'tokamak'),
    outputs='S_phi',
)
def sauter_plasma_cross_sectional_surface(
    a: float,
    kappa: float, 
    delta: float, 
    squareness: float) -> float:
    """
    Surface of the plasma cross-section in the radial and poloidal direction
    See :cite:`sauter_geometric_2016`.
    """
    # theta_07 reported in Sauter 2016 (:cite:`sauter_geometric_2016`) has been multiplied and divided by 1+sqrt(1+8*xi) to avoid division by 0
    theta_07 = np.arcsin(0.7) - 2 * squareness / (1 + np.sqrt(1 + 8 * squareness**2))
    # using the analytical formula for w_07, related to the radial width of the plasma shape at 70% of the maximum height
    w_07 = np.cos(theta_07 - squareness * np.sin(2 * theta_07)) / np.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    S_phi = math.pi * a**2 * kappa * (1 + 0.52 * (w_07 - 1))
    return  S_phi


@relation(
    name="Tokamak plasma poloidal length (PROCESS squareness)",
    tags=("geometry", "tokamak", "process"),
    outputs="L_p",
)
def process_sauter_poloidal_length(
    rminor: float,
    kappa: float,
    triang: float,
    plasma_square: float,
) -> float:
    """Poloidal length around the plasma cross-section (Sauter formula with
    PROCESS's simplified ``w07 = squareness + 1``, assuming top-down symmetry;
    fusdb's default relation computes w07 from Sauter's theta_07 fit instead).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - O. Sauter, Fusion Engineering and Design 112, 633-645 (2016)
    """
    # CHECK
    w07 = plasma_square + 1
    return (
        2.0e0
        * np.pi
        * rminor
        * (1 + 0.55 * (kappa - 1))
        * (1 + 0.08 * triang**2)
        * (1 + 0.2 * (w07 - 1))
    )


@relation(
    name="Tokamak plasma cross-sectional surface (PROCESS squareness)",
    tags=("geometry", "tokamak", "process"),
    outputs="S_phi",
)
def process_sauter_cross_sectional_surface(
    rminor: float,
    kappa: float,
    plasma_square: float,
) -> float:
    """Plasma cross-sectional area (Sauter formula with PROCESS's simplified
    ``w07 = squareness + 1``; fusdb's default relation computes w07 from
    Sauter's theta_07 fit instead).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - O. Sauter, Fusion Engineering and Design 112, 633-645 (2016)
    """
    # CHECK
    w07 = plasma_square + 1
    return np.pi * rminor**2 * kappa * (1 + 0.52 * (w07 - 1))


def sauter_cross_section_points(
    R: float,
    a: float,
    *,
    kappa: float,
    delta: float,
    squareness: float = 0.0,
    n: int = 256,
) -> tuple[list[float], list[float]]:
    """Return Sauter-style (R, Z) points for a tokamak plasma cross-section."""
    if n < 8:
        raise ValueError("n must be >= 8 for a meaningful cross-section")

    two_pi = 2.0 * math.pi
    r_vals: list[float] = []
    z_vals: list[float] = []
    for i in range(n):
        theta = two_pi * i / (n - 1)
        angle = theta + delta * np.sin(theta) - squareness * np.sin(2.0 * theta)
        r_vals.append(R + a * np.cos(angle))
        z_vals.append(kappa * a * np.sin(theta + squareness * np.sin(2.0 * theta)))
    return r_vals, z_vals
