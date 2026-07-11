"""Basic plasma shape relations."""

import numpy as np

from fusdb.relation import relation


@relation(
    name='Major radius',
    tags=('geometry',),
    outputs='R',
)
def major_radius(R_max: float, R_min: float) -> float:
    """Return the geometric major radius from inboard/outboard extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (R_max + R_min) / 2


@relation(
    name='minor radius',
    tags=('geometry',),
    outputs='a',
)
def minor_radius(R_max: float, R_min: float) -> float:
    """Return the geometric minor radius from inboard/outboard major radii extents.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (R_max - R_min) / 2


@relation(
    name='Aspect ratio',
    tags=('geometry',),
    
    outputs='A',
)
def aspect_ratio(R: float, a: float) -> float:
    """Return aspect ratio from major and minor radius.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return R / a


@relation(
    name='Inverse aspect ratio',
    tags=('geometry',),
    
    outputs='eps',
)
def aspect_ratio_relation(A: float) -> float:
    """Aspect ratio should be the inverse of the inverse aspect ratio"""
    return 1 / A


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


@relation(
    name='Elongation 95%',
    tags=('geometry', 'tokamak'),
    outputs='kappa_95',
)
def elongation_95(kappa: float) -> float:
  
    return kappa / 1.12


def triangularity(delta_top: float, delta_bottom: float) -> float:
    """Return triangularity from triangularity at the top and bottom of the shape.
    Formula reported in :cite:`sauter_geometric_2016`."""
    return (delta_top + delta_bottom) / 2


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


@relation(
    name='IPB elongation from volume',
    tags=('geometry',),
    
    outputs='kappa_ipb',
)
def kappa_ipb_from_volume(V_p: float, R: float, a: float) -> float:
    """Return IPB-specific elongation from volume and radii."""
    pi = np.pi
    return V_p / (2 * pi**2 * R * a**2)
