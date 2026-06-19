"""Spherical tokamak shape relations."""

from fusdb import relation


@relation(
    name='ST elongation vs aspect ratio',
    tags=('geometry', 'spherical_tokamak'),
    outputs='kappa',
)
def st_elongation_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak elongation from aspect ratio."""
    return 0.95 * (1.9 + 1.9 / (A ** 1.4))


@relation(
    name='ST triangularity vs aspect ratio',
    tags=('geometry', 'spherical_tokamak'),
    outputs='delta',
)
def st_triangularity_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak triangularity from aspect ratio."""
    return 0.53 * (1 + 0.77 * (1 / A) ** 3) / 1.50
