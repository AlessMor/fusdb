"""Plasma geometry relations defined once and solved generically."""

from __future__ import annotations

import math
import sympy as sp

from fusdb.relation_class import Relation_decorator as Relation

########################################################################################################################
#                                                                                                 RADII and ASPECT RATIO
########################################################################################################################
@Relation(
    name="Major radius",
    output="R",
    tags=("geometry",))
def major_radius(R_max: float, R_min: float) -> float:
    """Return the geometric major radius from inboard/outboard extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (R_max + R_min) / 2

########################################
@Relation(
    name="minor radius",
    output="a",
    tags=("geometry",))
def minor_radius(R_max: float, R_min: float) -> float:
    """Return the geometric minor radius from inboard/outboard major radii extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (R_max - R_min) / 2

########################################
@Relation(
    name="Aspect ratio",
    output="A",
    tags=("geometry",),
    constraints=("a != 0"))
def aspect_ratio(R: float, a: float) -> float:
    """Return aspect ratio from major and minor radius.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return R / a

########################################
@Relation(
    name="Inverse aspect ratio",
    output="eps",
    tags=("geometry",),
    constraints=("R != 0"))
def inverse_aspect_ratio(R: float, a: float) -> float:
    """Return inverse aspect ratio from major and minor radius.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return a / R

########################################
@Relation(
    name="Inverse aspect ratio from aspect ratio",
    output="eps",
    tags=("geometry",),
    constraints=("A != 0"))
def aspect_ratio_relation(A: float) -> float:
    """Aspect ratio should be the inverse of the inverse aspect ratio"""
    return 1 / A

########################################################################################################################
#                                                                               ELONGATION, TRIANGULARITY and SQUARENESS
########################################################################################################################

# TODO(med): differentiate between kappa_sep = (Zmax - Zmin)/(Rmax - Rmin) and kappa_areal = S_p/(pi*a**2)

########################################
@Relation(
    name="Elongation",
    output="kappa",
    tags=("geometry",),
    constraints=("R_max - R_min != 0","Z_max - Z_min != 0"),)
def elongation(Z_max: float, Z_min: float, R_max: float, R_min: float) -> float:
    """Return elongation from vertical and horizontal extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (Z_max - Z_min) / (R_max - R_min)
# TODO(med): check if "95" relations are valid for L,I and H-modes
########################################
@Relation(
    name="Elongation 95%",
    output="kappa",
    tags=("geometry", "tokamak"),
    initial_guesses={"kappa": lambda v: 1.12 * v["kappa_95"], "kappa_95": lambda v: v["kappa"] / 1.12},
)
def elongation_95(kappa_95: float) -> float:
    """Return core elongation from kappa_95.
    N.A. Uckan and ITER Physics Group, ITER Physics Design Guidelines: 1989, ITER Documentation Series, No. 10, IAEA/ITER/DS/10 (1990)
    """
    return 1.12 * kappa_95
# TODO(med): check if "95" relations are valid for L,I and H-modes

########################################
# TODO(low): add as relation. to do so it's necessary also to define delta_top = (R-R(Z=Zmax))/a and delta_bottom = (R-R(Z=Zmin))/a
def triangularity(delta_top: float, delta_bottom: float) -> float:
    """Return triangularity from triangularity at the top and bottom of the shape.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (delta_top + delta_bottom) / 2

########################################
@Relation(
    name="Triangularity 95%",
    output="delta",
    tags=("geometry", "tokamak"),
    initial_guesses={"delta": lambda v: 1.5 * v["delta_95"], "delta_95": lambda v: v["delta"] / 1.5},
)
def triangularity_95(delta_95: float) -> float:
    """Return core triangularity from delta_95.
    N.A. Uckan and ITER Physics Group, ITER Physics Design Guidelines: 1989, ITER Documentation Series, No. 10, IAEA/ITER/DS/10 (1990)
    """
    return 1.5 * delta_95

########################################
@Relation(
    name="IPB elongation from volume",
    output="kappa_ipb",
    tags=("geometry",),
    constraints=("R != 0", "a != 0"),
)
def kappa_ipb_from_volume(V_p: float, R: float, a: float) -> float:
    """Return IPB-specific elongation from volume and radii."""
    return V_p / (2 * sp.pi**2 * R * a**2)

#TODO(low): no geometry available for stellarators yet... Most papers do not give enough info to use complex relations (Henneberg, Boozer,...). maybe define a formula for "effective plasma volume/surface" for stellarators (V_p = 2π^2*R*a^2)

# Configuration-specific geometry guidance (simplified from PROCESS/STAR/ITER sources).
# For spherical tokamaks, see Menard et al., Nucl. Fusion 2016 and PROCESS Issue #1439/#1086.
########################################
@Relation(
    name="ST elongation vs aspect ratio",
    output="kappa",
    tags=("geometry", "sphericaltokamak"),
    initial_guesses={"kappa": lambda v: 2.0},
)
def st_elongation_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak elongation from aspect ratio."""
    return 0.95 * (1.9 + 1.9 / (A ** 1.4))
########################################
@Relation(
    name="ST triangularity vs aspect ratio",
    output="delta",
    tags=("geometry", "sphericaltokamak"),
    initial_guesses={"delta": lambda v: 0.3},
)
def st_triangularity_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak triangularity from aspect ratio."""
    return 0.53 * (1 + 0.77 * (1 / A) ** 3) / 1.50

########################################################################################################################
#                                                                                      VOLUMES, SURFACES, CROSS-SECTIONS
########################################################################################################################

########################################
@Relation(
    name = "Tokamak plasma poloidal length",
    output="L_p",
    tags=("geometry", "tokamak"),
)
def sauter_plasma_cross_sectional_length(
    a: float,
    kappa: float,
    delta: float,
    squareness: float,
):
    """
    Poloidal length around the plasma cross-section.
    See :cite:`sauter_geometric_2016`.
    """
    # theta_07 reported in Sauter 2016 (:cite:`sauter_geometric_2016`) has been multiplied and divided by 1+sqrt(1+8*xi) to avoid division by 0
    theta_07 = -2 * squareness / (1 + sp.sqrt(1 + 8 * squareness**2))
    # using the analytical formula for w_07, related to the radial width of the plasma shape at 70% of the maximum height
    w_07 = sp.cos(theta_07 - squareness * sp.sin(2 * theta_07)) / sp.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    L_p = 2 * math.pi * a * (1 + 0.55 * (kappa - 1)) * (1 + 0.08 * delta**2) * (1 + 0.2 * (w_07 - 1))
    return  L_p

########################################
@Relation(
    name="Tokamak plasma surface",
    output="A_p",
    tags=("geometry", "tokamak"),
)
def sauter_plasma_surface(
    R: float,
    delta: float,
    eps: float,
    L_p: float,
):
    """
    Surface area around the LCFS in the toroidal and poloidal direction
    See :cite:`sauter_geometric_2016`.
    """
    A_p = 2 * math.pi * R * (1 - 0.32 * delta * eps) * L_p
    return A_p

########################################
@Relation(
    name="Tokamak plasma volume",
    output="V_p",
    tags=("geometry", "tokamak"),
)
def sauter_plasma_volume(
    R: float,
    delta: float,
    eps: float,
    S_phi: float,
):
    """
    Plasma Volume inside the LCFS
    See :cite:`sauter_geometric_2016`.
    """
    V_p = 2 * math.pi * R * (1 - 0.25 * delta * eps) * S_phi
    return V_p

########################################
@Relation(
    name="Tokamak plasma cross-sectional surface",
    output="S_phi",
    tags=("geometry", "tokamak"),
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
    theta_07 = sp.asin(0.7) - 2 * squareness / (1 + sp.sqrt(1 + 8 * squareness**2))
    # using the analytical formula for w_07, related to the radial width of the plasma shape at 70% of the maximum height
    w_07 = sp.cos(theta_07 - squareness * sp.sin(2 * theta_07)) / sp.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    S_phi = sp.pi * a**2 * kappa * (1 + 0.52 * (w_07 - 1))
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
        angle = theta + delta * math.sin(theta) - squareness * math.sin(2.0 * theta)
        r_vals.append(R + a * math.cos(angle))
        z_vals.append(kappa * a * math.sin(theta + squareness * math.sin(2.0 * theta)))
    return r_vals, z_vals

# TODO(med): add from cfspopcon "calc_plasma_poloidal_circumference"
