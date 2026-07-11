"""Plasma geometry from two intersecting arcs (PROCESS non-Sauter branch).

Ported from PROCESS ``process/models/physics/plasma_geometry.py``
(``plasma_angles_arcs`` + consumers). The plasma boundary is modelled by the
revolution of two intersecting arcs around the device centreline, appropriate
for plasmas with a separatrix. PROCESS's obsolete legacy module-level
functions (``surfa``/``perim``/``fvol``/``xsect0``) are algebraically
equivalent reformulations of these and were not ported.

All outputs are gated in variables.yaml: the Sauter-geometry relations remain
the fusdb defaults.
"""

import numpy as np

from fusdb.relation import relation


def _plasma_angles_arcs(a, kappa, triang):
    """Radii and half-angles of the inboard/outboard arcs (PROCESS
    ``plasma_angles_arcs``); reference F/MI/PJK/LOGBOOK14, p.42."""
    t = 1.0e0 - triang
    denomi = (kappa**2 - t**2) / (2.0e0 * t)
    thetai = np.arctan(kappa / denomi)
    xi = a * (denomi + 1.0e0 - triang)

    n = 1.0e0 + triang
    denomo = (kappa**2 - n**2) / (2.0e0 * n)
    thetao = np.arctan(kappa / denomo)
    xo = a * (denomo + 1.0e0 + triang)

    return xi, thetai, xo, thetao


@relation(
    name="Plasma poloidal perimeter from arcs",
    tags=("geometry", "tokamak", "process"),
    outputs="L_p",
)
def plasma_poloidal_perimeter(rminor: float, kappa: float, triang: float) -> float:
    """Calculate the plasma poloidal perimeter from the intersecting-arcs model.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    xi, thetai, xo, thetao = _plasma_angles_arcs(rminor, kappa, triang)
    return 2.0e0 * (xo * thetao + xi * thetai)


@relation(
    name="Plasma surface area from arcs",
    tags=("geometry", "tokamak", "process"),
    outputs="A_p",
)
def plasma_surface_area(rmajor: float, rminor: float, kappa: float, triang: float) -> float:
    """Calculate the plasma surface area (inboard + outboard) from the
    intersecting-arcs model.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    xi, thetai, xo, thetao = _plasma_angles_arcs(rminor, kappa, triang)
    fourpi = 4.0e0 * np.pi

    rc = rmajor - rminor + xi
    xsi = fourpi * xi * (rc * thetai - xi * np.sin(thetai))

    rc = rmajor + rminor - xo
    xso = fourpi * xo * (rc * thetao + xo * np.sin(thetao))

    return xsi + xso


@relation(
    name="Plasma outboard surface area from arcs",
    tags=("geometry", "tokamak", "process"),
    outputs="A_p_out",
)
def plasma_outboard_surface_area(rmajor: float, rminor: float, kappa: float, triang: float) -> float:
    """Calculate the outboard plasma surface area from the intersecting-arcs
    model (PROCESS ``a_plasma_surface_outboard``, needed by blanket/divertor
    models).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    _xi, _thetai, xo, thetao = _plasma_angles_arcs(rminor, kappa, triang)
    rc = rmajor + rminor - xo
    return 4.0e0 * np.pi * xo * (rc * thetao + xo * np.sin(thetao))


@relation(
    name="Plasma volume from arcs",
    tags=("geometry", "tokamak", "process"),
    outputs="V_p",
)
def plasma_volume(rmajor: float, rminor: float, kappa: float, triang: float) -> float:
    """Calculate the plasma volume from the intersecting-arcs model.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS multiplies this by the input fudge factor ``f_vol_plasma`` in its
    orchestration; the factor is not part of the geometric formula and is not
    ported.
    """
    # CHECK
    xi, thetai, xo, thetao = _plasma_angles_arcs(rminor, kappa, triang)
    third = 1.0e0 / 3.0e0

    rc = rmajor - rminor + xi
    vin = (
        2.0
        * np.pi
        * xi
        * (
            rc**2 * np.sin(thetai)
            - rc * xi * thetai
            - 0.5e0 * rc * xi * np.sin(2.0e0 * thetai)
            + xi * xi * np.sin(thetai)
            - third * xi * xi * (np.sin(thetai)) ** 3
        )
    )

    rc = rmajor + rminor - xo
    vout = (
        2.0
        * np.pi
        * xo
        * (
            rc**2 * np.sin(thetao)
            + rc * xo * thetao
            + 0.5e0 * rc * xo * np.sin(2.0e0 * thetao)
            + xo * xo * np.sin(thetao)
            - third * xo * xo * (np.sin(thetao)) ** 3
        )
    )

    return vout - vin


@relation(
    name="Plasma cross-section from arcs",
    tags=("geometry", "tokamak", "process"),
    outputs="S_phi",
)
def plasma_cross_section(rminor: float, kappa: float, triang: float) -> float:
    """Calculate the plasma poloidal cross-sectional area from the
    intersecting-arcs model.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    xi, thetai, xo, thetao = _plasma_angles_arcs(rminor, kappa, triang)
    return xo**2 * (thetao - np.cos(thetao) * np.sin(thetao)) + xi**2 * (
        thetai - np.cos(thetai) * np.sin(thetai)
    )
