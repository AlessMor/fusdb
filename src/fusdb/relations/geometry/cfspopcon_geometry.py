"""Plasma geometry in cfspopcon's closed form.

Ported from cfspopcon ``formulas/geometry/analytical.py``. These are a distinct
shape model from fusdb's Sauter defaults, not a reparametrisation of them: they
are closed forms in ``(R, eps, kappa_areal)`` alone, with no triangularity and
no squareness dependence at all. cfspopcon's own note on the volume explains
why -- "delta=1.0 is assumed since this was found to give a closer match to 2D
equilibria from FreeGS" -- so the triangularity is baked into the coefficients
rather than carried as a variable.

The difference is not small and not removable by choosing a different delta. At
the SPARC PRD point (R 1.85, eps 0.3081, kappa_areal 1.75) this gives
``A_p = 55.536``, reproducing cfspopcon's reference dataset exactly, while
fusdb's Sauter form gives 60.39 at the separatrix triangularity 0.54 and 57.90
even at delta = 1.0.

All outputs are gated in variables.yaml: the Sauter relations remain the fusdb
defaults, and these are selected by name (the cfspopcon SPARC comparison does).
"""

import numpy as np

from fusdb.relation import relation


@relation(
    name="Plasma volume (cfspopcon)",
    tags=("geometry", "tokamak"),
    outputs="V_p",
)
def calc_plasma_volume(R: float, eps: float, kappa_areal: float) -> float:
    """Plasma volume inside an up-down symmetrical LCFS.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        2.0 * np.pi * R**3.0 * eps**2.0 * kappa_areal
        * (np.pi - (np.pi - 8.0 / 3.0) * eps)
    )


@relation(
    name="Plasma surface area (cfspopcon)",
    tags=("geometry", "tokamak"),
    outputs="A_p",
)
def calc_plasma_surface_area(R: float, eps: float, kappa_areal: float) -> float:
    """Plasma surface area inside the LCFS.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        2.0 * np.pi * (R**2.0) * eps * kappa_areal
        * (np.pi + 2.0 - (np.pi - 2.0) * eps)
    )


@relation(
    name="Plasma poloidal circumference (cfspopcon)",
    tags=("geometry", "tokamak"),
    outputs="L_p",
)
def calc_plasma_poloidal_circumference(a: float, kappa_areal: float) -> float:
    """Plasma poloidal circumference at the LCFS.

    fusdb's Sauter ``Tokamak plasma poloidal length`` is this same leading form
    with additional ``(1 + 0.08 delta^2)`` and squareness factors; cfspopcon
    drops both.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return 2.0 * np.pi * a * (1.0 + 0.55 * (kappa_areal - 1.0))
