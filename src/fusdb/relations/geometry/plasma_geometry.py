"""Plasma geometry relations defined once and solved generically."""

from __future__ import annotations

import numpy as np
import math

from fusdb import relation


########################################################################################################################
#                                                                                                 RADII and ASPECT RATIO
########################################################################################################################
@relation(
    name='Major radius',
    tags=('geometry',),
    outputs='R',
)
def major_radius(R_max: float, R_min: float) -> float:
    """Return the geometric major radius from inboard/outboard extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (R_max + R_min) / 2

########################################
@relation(
    name='minor radius',
    tags=('geometry',),
    outputs='a',
)
def minor_radius(R_max: float, R_min: float) -> float:
    """Return the geometric minor radius from inboard/outboard major radii extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (R_max - R_min) / 2

########################################
@relation(
    name='Aspect ratio',
    tags=('geometry',),
    
    outputs='A',
)
def aspect_ratio(R: float, a: float) -> float:
    """Return aspect ratio from major and minor radius.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return R / a

########################################
@relation(
    name='Inverse aspect ratio',
    tags=('geometry',),
    
    outputs='eps',
)
def aspect_ratio_relation(A: float) -> float:
    """Aspect ratio should be the inverse of the inverse aspect ratio"""
    return 1 / A

########################################################################################################################
#                                                                               ELONGATION, TRIANGULARITY and SQUARENESS
########################################################################################################################

# TODO(med): differentiate between kappa_sep = (Zmax - Zmin)/(Rmax - Rmin) and kappa_areal = S_p/(pi*a**2)

########################################
@relation(
    name='Elongation',
    tags=('geometry',),
    constraints=('R_max > R_min', 'Z_max > Z_min'),
    outputs='kappa',
)
def elongation(Z_max: float, Z_min: float, R_max: float, R_min: float) -> float:
    """Return elongation from vertical and horizontal extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (Z_max - Z_min) / (R_max - R_min)
# TODO(med): check if "95" relations are valid for L,I and H-modes
########################################
@relation(
    name='Elongation 95%',
    tags=('geometry', 'tokamak'),
    outputs='kappa_95',
)
def elongation_95(kappa: float) -> float:
  
    return kappa / 1.12
# TODO(med): check if "95" relations are valid for L,I and H-modes

########################################
# TODO(low): add as relation. to do so it's necessary also to define delta_top = (R-R(Z=Zmax))/a and delta_bottom = (R-R(Z=Zmin))/a
def triangularity(delta_top: float, delta_bottom: float) -> float:
    """Return triangularity from triangularity at the top and bottom of the shape.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (delta_top + delta_bottom) / 2

########################################
@relation(
    name='Triangularity 95%',
    tags=('geometry', 'tokamak'),
    outputs='delta',
)
def triangularity_95(delta_95: float) -> float:
    """Return core triangularity from delta_95.
    N.A. Uckan and ITER Physics Group, ITER Physics Design Guidelines: 1989, ITER Documentation Series, No. 10, IAEA/ITER/DS/10 (1990)
    """
    return 1.5 * delta_95

########################################
@relation(
    name='IPB elongation from volume',
    tags=('geometry',),
    
    outputs='kappa_ipb',
)
def kappa_ipb_from_volume(V_p: float, R: float, a: float) -> float:
    """Return IPB-specific elongation from volume and radii."""
    pi = np.pi
    return V_p / (2 * pi**2 * R * a**2)

#TODO(low): no geometry available for stellarators yet... Most papers do not give enough info to use complex relations (Henneberg, Boozer,...). maybe define a formula for "effective plasma volume/surface" for stellarators (V_p = 2π^2*R*a^2)

# Configuration-specific geometry guidance (simplified from PROCESS/STAR/ITER sources).
# For spherical tokamaks, see Menard et al., Nucl. Fusion 2016 and PROCESS Issue #1439/#1086.
########################################
@relation(
    name='ST elongation vs aspect ratio',
    tags=('geometry', 'spherical_tokamak'),
    outputs='kappa',
)
def st_elongation_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak elongation from aspect ratio."""
    return 0.95 * (1.9 + 1.9 / (A ** 1.4))
########################################
@relation(
    name='ST triangularity vs aspect ratio',
    tags=('geometry', 'spherical_tokamak'),
    outputs='delta',
)
def st_triangularity_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak triangularity from aspect ratio."""
    return 0.53 * (1 + 0.77 * (1 / A) ** 3) / 1.50

########################################################################################################################
#                                                                                      VOLUMES, SURFACES, CROSS-SECTIONS
########################################################################################################################

########################################
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

########################################
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

########################################
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

########################################
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

########################################
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

# TODO(med): add from cfspopcon "calc_plasma_poloidal_circumference"

########################################################################################################################
#  cfspopcon ports (UNDECORATED scaffolds) — ported verbatim from cfspopcon for review.
#  Source: cfspopcon/formulas/geometry/analytical.py and plasma_current/safety_factor.py
#  These are NOT yet fusdb @relation's: review the formula, map variable names/units to
#  the fusdb registry (e.g. areal_elongation->kappa, magnetic_field_on_axis->B0,
#  inverse_aspect_ratio->eps, plasma_current->I_p in A vs MA), then add @relation to activate.
########################################################################################################################


# TODO(cfspopcon): activate as a fusdb relation (separatrix_elongation output).
def calc_separatrix_elongation_from_areal_elongation(areal_elongation, elongation_ratio_sep_to_areal):
    """cfspopcon: separatrix_elongation = areal_elongation * elongation_ratio_sep_to_areal."""
    return areal_elongation * elongation_ratio_sep_to_areal


# TODO(cfspopcon): activate as a fusdb relation (separatrix_triangularity output).
def calc_separatrix_triangularity_from_triangularity95(triangularity_psi95, triangularity_ratio_sep_to_psi95):
    """cfspopcon: separatrix_triangularity = triangularity_psi95 * triangularity_ratio_sep_to_psi95."""
    return triangularity_psi95 * triangularity_ratio_sep_to_psi95


# TODO(cfspopcon): activate as a fusdb relation (vertical_minor_radius output).
def calc_vertical_minor_radius_from_elongation_and_minor_radius(minor_radius, separatrix_elongation):
    """cfspopcon: vertical_minor_radius = minor_radius * separatrix_elongation."""
    return minor_radius * separatrix_elongation


# TODO(cfspopcon): activate as a fusdb relation (elongation_psi95 output).
def calc_elongation_at_psi95_from_areal_elongation(areal_elongation, elongation_ratio_areal_to_psi95):
    """cfspopcon: elongation_psi95 = areal_elongation / elongation_ratio_areal_to_psi95."""
    return areal_elongation / elongation_ratio_areal_to_psi95


# TODO(cfspopcon): activate as a fusdb relation (f_shaping output).
def calc_f_shaping_for_qstar(inverse_aspect_ratio, areal_elongation, triangularity_psi95):
    """cfspopcon: shaping function for q_star (ITER Physics Basis Ch.1 Eq. A-11)."""
    return ((1.0 + areal_elongation**2.0 * (1.0 + 2.0 * triangularity_psi95**2.0 - 1.2 * triangularity_psi95**3.0)) / 2.0) * (
        (1.17 - 0.65 * inverse_aspect_ratio) / (1.0 - inverse_aspect_ratio**2.0) ** 2.0
    )


# TODO(cfspopcon): activate as a fusdb relation (q_star output). plasma_current in MA.
def calc_q_star_from_plasma_current(magnetic_field_on_axis, major_radius, inverse_aspect_ratio, plasma_current, f_shaping):
    """cfspopcon: analytical edge safety factor q_star (ITER Physics Basis Ch.1)."""
    return (
        5.0 * (inverse_aspect_ratio * major_radius) ** 2.0 * magnetic_field_on_axis / (plasma_current * major_radius) * f_shaping
    )


# TODO(cfspopcon): activate as a fusdb relation (cylindrical_safety_factor output).
#   Uses kappa_95/delta_95; mu_0 below is the vacuum permeability [T*m/A].
def calc_cylindrical_edge_safety_factor(
    major_radius, minor_radius, elongation_psi95, triangularity_psi95, magnetic_field_on_axis, plasma_current
):
    """cfspopcon: edge safety factor following SepOS (Eich 2021 Eq. K.6). plasma_current in A."""
    mu_0 = 1.25663706212e-6
    shaping_correction = np.sqrt((1.0 + elongation_psi95**2 * (1.0 + 2.0 * triangularity_psi95**2 - 1.2 * triangularity_psi95**3)) / 2.0)
    poloidal_circumference = 2.0 * np.pi * minor_radius * shaping_correction
    average_B_pol = mu_0 * plasma_current / poloidal_circumference
    return magnetic_field_on_axis / average_B_pol * minor_radius / major_radius * shaping_correction
