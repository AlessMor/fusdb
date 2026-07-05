"""Plasma shape scalings/fits from the PROCESS geometry-model dispatcher.

Ported from PROCESS ``process/models/physics/plasma_geometry.py`` ``run()``
(``i_plasma_geometry`` switch), split into one relation per model. PROCESS's
paired ``*_X_POINT`` / ``*_95`` enum entries are the same equation solved for
the other variable; fusdb relations are adirectional, so each pair collapses
into a single relation. The IPDG89 kappa/1.12 and triang/1.5 conversions and
the Menard-2016 elongation scaling already exist in fusdb and were not
re-imported.

All producers of kappa/delta/kappa_95 here are gated in variables.yaml: the
pre-existing fusdb relations remain the defaults.
"""

import numpy as np

from fusdb import relation


@relation(
    name="STAR elongation from aspect ratio",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="kappa",
)
def star_elongation_from_aspect_ratio(eps: float) -> float:
    """Return elongation from the STAR-code spherical-tokamak scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - J D Galambos, STAR Code : Spherical Tokamak Analysis and Reactor Code,
          unpublished internal Oak Ridge document
    """
    # CHECK
    return 2.05e0 * (1.0e0 + 0.44e0 * eps**2.1e0)


@relation(
    name="STAR triangularity from aspect ratio",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="delta",
)
def star_triangularity_from_aspect_ratio(eps: float) -> float:
    """Return triangularity from the STAR-code spherical-tokamak scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.53e0 * (1.0e0 + 0.77e0 * eps**3)


@relation(
    name="STAR minimum q95 from aspect ratio",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="q95_min",
)
def star_minimum_q95_from_aspect_ratio(eps: float) -> float:
    """Return the minimum safe q95 from the STAR-code spherical-tokamak
    scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 3.0e0 * (1.0e0 + 2.6e0 * eps**2.8e0)


@relation(
    name="Zohm ITER elongation scaling",
    tags=("geometry", "tokamak", "process"),
    outputs="kappa",
)
def zohm_iter_elongation(aspect: float, fkzohm: float) -> float:
    """Return elongation from the Zohm et al. ITER aspect-ratio scaling with
    adjustment factor ``fkzohm``.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - H. Zohm et al, On the Physics Guidelines for a Tokamak DEMO,
          FTP/3-3, Proc. IAEA Fusion Energy Conference, October 2012, San Diego
    """
    # CHECK
    return fkzohm * np.minimum(2.0e0, 1.5e0 + 0.5e0 / (aspect - 1.0e0))


@relation(
    name="MAST elongation fit",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="kappa",
)
def mast_elongation_fit(kappa95: float) -> float:
    """Return separatrix elongation from the 95% flux-surface elongation using
    the fit to MAST data (PROCESS Issue #1086). Adirectional: also yields
    kappa95 from kappa (PROCESS ``MAST_DATA_X_POINT``).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.91300e0 * kappa95 + 0.38654e0


@relation(
    name="MAST triangularity fit",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="delta",
)
def mast_triangularity_fit(triang95: float) -> float:
    """Return separatrix triangularity from the 95% flux-surface triangularity
    using the fit to MAST data (PROCESS Issue #1086). Adirectional: also
    yields triang95 from triang (PROCESS ``MAST_DATA_X_POINT``).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.77394e0 * triang95 + 0.18515e0


@relation(
    name="FIESTA elongation fit",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="kappa",
)
def fiesta_elongation_fit(kappa95: float) -> float:
    """Return separatrix elongation from the 95% flux-surface elongation using
    the fit to FIESTA runs (PROCESS Issue #1086). Adirectional: also yields
    kappa95 from kappa (PROCESS ``FIESTA_RUNS_X_POINT`` / ``STAR_FIESTA``).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.90698e0 * kappa95 + 0.39467e0


@relation(
    name="FIESTA triangularity fit",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="delta",
)
def fiesta_triangularity_fit(triang95: float) -> float:
    """Return separatrix triangularity from the 95% flux-surface triangularity
    using the fit to FIESTA runs (PROCESS Issue #1086). Adirectional: also
    yields triang95 from triang (PROCESS ``FIESTA_RUNS_X_POINT`` /
    ``STAR_FIESTA``).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 1.3799e0 * triang95 + 0.048306e0


@relation(
    name="Elongation from internal inductance scaling",
    tags=("geometry", "tokamak", "process"),
    outputs="kappa",
)
def elongation_from_internal_inductance(aspect: float, ind_plasma_internal_norm: float) -> float:
    """Return elongation from the aspect ratio and the normalized plasma
    internal inductance li(3) (PROCESS ``INDUCTANCE_SCALING_X_POINT``).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return (1.09e0 + 0.26e0 / ind_plasma_internal_norm) * (1.5e0 / aspect) ** 0.4e0


@relation(
    name="CREATE EU-DEMO elongation 95 fit",
    tags=("geometry", "tokamak", "process"),
    outputs="kappa_95",
)
def create_eu_demo_elongation_95_fit(aspect: float, m_s_limit: float) -> float:
    """Return the 95% flux-surface elongation from the aspect ratio and the
    vertical-stability margin, based on a fit to CREATE data for an EU-DEMO
    like machine (aspect ratio 2.6 - 3.6; PROCESS Issues #1399/#1648),
    including PROCESS's high-elongation corner correction.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    a = 3.68436807e0
    b = -0.27706527e0
    c = 0.87040251e0
    d = -18.83740952e0
    e = -0.27267618e0
    f = 20.5141261e0

    kappa95 = (
        -d
        - c * aspect
        - np.sqrt(
            (c**2.0e0 - 4.0e0 * a * b) * aspect**2.0e0
            + (2.0e0 * d * c - 4.0e0 * a * e) * aspect
            + d**2.0e0
            - 4.0e0 * a * f
            + 4.0e0 * a * m_s_limit
        )
    ) / (2.0e0 * a)

    if kappa95 > 1.77:
        ratio = 1.77 / kappa95
        corner_fudge = 0.3 * (kappa95 - 1.77) / ratio
        kappa95 = kappa95 ** (ratio) + corner_fudge

    return kappa95


@relation(
    name="Menard 1997 elongation from aspect ratio",
    tags=("geometry", "spherical_tokamak", "process"),
    outputs="kappa",
)
def menard_1997_elongation(aspect: float) -> float:
    """Return elongation from the Menard 1997 aspect-ratio scaling (maximum
    controllable kappa at constant li(3)).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - J.E. Menard et al 1997 Nucl. Fusion 37 595
    """
    # CHECK
    return 2.93e0 * (1.8e0 / aspect) ** 0.4e0
