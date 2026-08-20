"""Plasma magnetic-field relations.

Ported from PROCESS ``process/models/physics/plasma_fields.py``. The
surface-averaged poloidal field here is the conventional-tokamak Ampere's-law
branch; the Peng/STAR spherical-tokamak branch depends on ``plascar_bpol``
(PROCESS ``plasma_current.py``) and is ported with that module instead.
"""

from typing import Any

import numpy as np

from fusdb.numerics import volume_average
from fusdb.relation import relation
from fusdb.registry import MU0


@relation(
    name="Surface-averaged poloidal field from plasma current",
    tags=("plasma", "process"),
    outputs="B_p",
)
def calculate_surface_averaged_poloidal_field(plasma_current: float, len_plasma_poloidal: float) -> float:
    """Calculate the surface-averaged poloidal field (<Bp(a)>) from the plasma
    current using Ampere's law over the plasma poloidal cross-section.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    plasma_current :
        Plasma current [A]
    len_plasma_poloidal :
        Plasma poloidal perimeter [m]

    Returns
    -------
    :
        Surface-averaged poloidal field [T]

    References
    ----------
        - J D Galambos, STAR Code : Spherical Tokamak Analysis and Reactor Code,
          unpublished internal Oak Ridge document
    """
    # CHECK
    return MU0 * plasma_current / len_plasma_poloidal


@relation(
    name="Toroidal field at plasma inboard midplane",
    tags=("plasma", "process"),
    outputs="B_t_in_mid",
)
def calculate_plasma_inboard_toroidal_field(
    b_plasma_toroidal_on_axis: float, rmajor: float, rminor: float
) -> float:
    """Calculate the toroidal field at the plasma inboard midplane (BT(R0-a)).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    b_plasma_toroidal_on_axis :
        Toroidal field on axis [T]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]

    Returns
    -------
    :
        Toroidal field at the plasma inboard midplane [T]
    """
    # CHECK
    return rmajor * b_plasma_toroidal_on_axis / (rmajor - rminor)


@relation(
    name="Total magnetic field",
    tags=("plasma", "process"),
    outputs="B_total",
)
def calculate_total_magnetic_field(b_plasma_toroidal_on_axis: float, B_p: float) -> float:
    """Calculate the total magnetic field from the on-axis toroidal field and
    the surface-averaged poloidal field, matching how PROCESS assembles
    ``b_plasma_total``.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    b_plasma_toroidal_on_axis :
        Toroidal field on axis [T]
    B_p :
        Surface-averaged poloidal field [T]

    Returns
    -------
    :
        Total magnetic field [T]
    """
    # CHECK
    return np.sqrt(b_plasma_toroidal_on_axis**2 + B_p**2)


@relation(name="Tokamak magnetic field B2.5 moment", tags=("tokamak", "geometry", "power_balance"), outputs="G_B25")
def tokamak_magnetic_field_b25_moment(B: Any, B0: Any, rho: Any, w_V: Any = None) -> Any:
    """Normalized volume moment <|B/B0|^2.5> used by cyclotron-loss models.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return volume_average(np.abs(np.asarray(B, dtype=float) / np.asarray(B0, dtype=float)) ** 2.5, rho, weight=w_V)
