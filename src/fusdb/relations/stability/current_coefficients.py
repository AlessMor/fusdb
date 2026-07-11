"""Plasma-current shaping coefficients (PROCESS ``i_plasma_current`` models).

Ported from PROCESS ``process/models/physics/plasma_current.py``. PROCESS forms
the plasma current as ``I_p = I_cyl * fq`` where the cylindrical current is
``I_cyl = (2*pi/mu0) * rminor^2 / (rmajor*q95) * B0``. Since ``2*pi/mu0 = 5e6``
(mu0 = 4*pi*1e-7), this is exactly fusdb's existing ``Edge safety factor
q_star`` relation (``qstar = 5*rminor^2*B0/(I_p[MA]*R) * f_shaping``) solved for
the current, with the PROCESS ``fq`` playing the role of fusdb's ``f_shaping``.

Each ``fq`` scaling is therefore imported here as an alternative producer of
``f_shaping`` (the existing ``qstar`` relation ties it back to ``I_p``), gated in
variables.yaml so the cfspopcon IPB default stays the default. The
``i_plasma_current`` enum dispatcher itself is not ported.

Not imported / skipped:
    * IPDG89 ``fq`` -- algebraically identical to the existing cfspopcon
      ``f_shaping`` (both are ITER Physics Basis Ch.1 Eq. A-11); PROCESS feeds
      kappa95/triang95 where cfspopcon feeds kappa/delta_95.
    * ITER "simple" scaling -- ``fq = 1`` (a trivial constant).
    * cylindrical-current helper -- it is the existing ``q_star`` relation.
"""

import numpy as np

from fusdb.relation import relation
from fusdb.registry import MU0


def _plascar_bpol(aspect, eps, kappa, delta):
    """Peng/STAR poloidal-field coefficients (PROCESS ``plascar_bpol``), used by
    the TART Peng plasma-current and poloidal-field scalings. Returns
    ``ff1, ff2, d1, d2``. Reference: Peng, Galambos & Shipe (1992)."""
    c1 = (kappa**2 / (1.0 + delta)) + delta
    c2 = (kappa**2 / (1.0 - delta)) - delta

    d1 = (kappa / (1.0 + delta)) ** 2 + 1.0
    d2 = (kappa / (1.0 - delta)) ** 2 + 1.0

    c1_aspect = ((c1 * eps) - 1.0) if aspect < c1 else (1.0 - (c1 * eps))

    y1 = np.sqrt(c1_aspect / (1.0 + eps)) * ((1.0 + delta) / kappa)
    y2 = np.sqrt((c2 * eps + 1.0) / (1.0 - eps)) * ((1.0 - delta) / kappa)

    h2 = (1.0 + (c2 - 1.0) * (eps / 2.0)) / np.sqrt((1.0 - eps) * (c2 * eps + 1.0))
    f2 = (d2 * (1.0 - delta) * eps) / ((1.0 - eps) * ((c2 * eps) + 1.0))
    g = (eps * kappa) / (1.0 - (eps * delta))
    ff2 = f2 * (g + 2.0 * h2 * np.arctan(y2))

    h1 = (1.0 + (1.0 - c1) * (eps / 2.0)) / np.sqrt((1.0 + eps) * c1_aspect)
    f1 = (d1 * (1.0 + delta) * eps) / ((1.0 + eps) * (c1 * eps - 1.0))

    if aspect < c1:
        ff1 = f1 * (g - h1 * np.log((1.0 + y1) / (1.0 - y1)))
    else:
        ff1 = -f1 * (-g + 2.0 * h1 * np.arctan(y1))

    return ff1, ff2, d1, d2


@relation(
    name="Plasma shaping function Peng analytic",
    tags=("plasma", "stability", "tokamak", "process"),
    outputs="f_shaping",
)
def current_coefficient_peng(eps: float, L_p: float, a: float) -> float:
    """Plasma-current shaping coefficient from the Peng analytic fit (STAR code).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        (1.22 - 0.68 * eps)
        / ((1.0 - eps * eps) ** 2)
        * (L_p / (2.0 * np.pi * a)) ** 2
    )


@relation(
    name="Plasma shaping function Todd I",
    tags=("plasma", "stability", "tokamak", "process"),
    outputs="f_shaping",
)
def current_coefficient_todd_i(eps: float, kappa_95: float, delta_95: float) -> float:
    """Plasma-current shaping coefficient from the first Todd empirical scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - D.C.Robinson and T.N.Todd, Plasma and Contr Fusion 28 (1986) 1181
    """
    # CHECK
    return (
        (1.0 + 2.0 * eps**2)
        * ((1.0 + kappa_95**2) / 2)
        * (
            1.24
            - 0.54 * kappa_95
            + 0.3 * (kappa_95**2 + delta_95**2)
            + 0.125 * delta_95
        )
    )


@relation(
    name="Plasma shaping function Todd II",
    tags=("plasma", "stability", "tokamak", "process"),
    outputs="f_shaping",
)
def current_coefficient_todd_ii(eps: float, kappa_95: float, delta_95: float) -> float:
    """Plasma-current shaping coefficient from the second Todd empirical scaling
    (first scaling with a high-elongation correction).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - D.C.Robinson and T.N.Todd, Plasma and Contr Fusion 28 (1986) 1181
    """
    # CHECK
    base_scaling = (
        (1.0 + 2.0 * eps**2)
        * ((1.0 + kappa_95**2) / 2)
        * (
            1.24
            - 0.54 * kappa_95
            + 0.3 * (kappa_95**2 + delta_95**2)
            + 0.125 * delta_95
        )
    )
    return base_scaling * (1.0 + (abs(kappa_95 - 1.2)) ** 3)


@relation(
    name="Plasma shaping function Connor-Hastie",
    tags=("plasma", "stability", "tokamak", "process"),
    outputs="f_shaping",
)
def current_coefficient_hastie(
    alphaj: float,
    alphap: float,
    B0: float,
    delta_95: float,
    eps: float,
    kappa_95: float,
    pres_plasma_on_axis: float,
) -> float:
    """Plasma-current shaping coefficient from the Connor-Hastie
    asymptotically-correct model.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - J.W.Connor and R.J.Hastie, Culham Lab Report CLM-M106 (1985)
        - T.C.Hender et.al., 'Physics Assessment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # CHECK
    rmu0 = MU0
    lamda = alphaj
    nu = alphap
    beta0 = 2.0 * rmu0 * pres_plasma_on_axis / (B0**2)

    lamp1 = 1.0 + lamda
    li = lamp1 / lamda * (lamp1 / lamda * np.log(lamp1) - 1.0)

    kap1 = kappa_95 + 1.0
    tr = kappa_95 * delta_95 / kap1**2
    er = (kappa_95 - 1.0) / kap1
    tprime = 2.0 * tr * lamp1 / (1.0 + 0.5 * lamda)
    eprime = er * lamp1 / (1.0 + lamda / 3.0)

    deltap = (0.5 * kap1 * eps * 0.5 * li) + (beta0 / (0.5 * kap1 * eps)) * lamp1**2 / (
        1.0 + nu
    )
    deltar = beta0 / 6.0 * (1.0 + 5.0 * lamda / 6.0 + 0.25 * lamda**2) + (
        0.5 * kap1 * eps
    ) ** 2 * 0.125 * (1.0 - (lamda**2) / 3.0)

    return (0.5 * kap1) ** 2 * (
        1.0
        + eps**2 * (0.5 * kap1) ** 2
        + 0.5 * deltap**2
        + 2.0 * deltar
        + 0.5 * (eprime**2 + er**2)
        + 0.5 * (tprime**2 + 4.0 * tr**2)
    )


@relation(
    name="Plasma shaping function Sauter",
    tags=("plasma", "stability", "tokamak", "process"),
    outputs="f_shaping",
)
def current_coefficient_sauter(eps: float, kappa: float, delta: float) -> float:
    """Plasma-current shaping coefficient from the Sauter model (allows negative
    triangularity, assumes zero squareness).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - O. Sauter, Fusion Engineering and Design 112, 633-645 (2016)
    """
    # CHECK
    w07 = 1.0  # zero squareness
    return (
        (4.1e6 / 5.0e6)
        * (1.0 + 1.2 * (kappa - 1.0) + 0.56 * (kappa - 1.0) ** 2)
        * (1.0 + 0.09 * delta + 0.16 * delta**2)
        * (1.0 + 0.45 * delta * eps)
        / (1.0 - 0.74 * eps)
        * (1.0 + 0.55 * (w07 - 1.0))
    )


@relation(
    name="Plasma shaping function FIESTA",
    tags=("plasma", "stability", "spherical_tokamak", "process"),
    outputs="f_shaping",
)
def current_coefficient_fiesta(eps: float, kappa: float, delta: float) -> float:
    """Plasma-current shaping coefficient from the FIESTA spherical-tokamak
    scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - S. Muldrew et al., Fusion Engineering and Design 154 (2020) 111530
    """
    # CHECK
    return 0.538 * (1.0 + 2.440 * eps**2.736) * kappa**2.154 * delta**0.060


@relation(
    name="Plasma current Peng TART scaling",
    tags=("plasma", "stability", "spherical_tokamak", "process"),
    outputs="I_p",
)
def plasma_current_peng(
    q95: float, A: float, eps: float, a: float, B0: float, kappa: float, delta: float
) -> float:
    """Plasma current from the Peng double-null divertor scaling for tight
    aspect-ratio tokamaks (STAR code). Unlike the other models this is not the
    cylindrical current times a shaping factor.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS returns the current in MA; fusdb's ``I_p`` is in A.

    References
    ----------
        - Peng, Galambos & Shipe (1992), Fusion Technology 21(3P2A), 1729-1738
    """
    # CHECK
    qbar = q95 * 1.3e0 * (1.0e0 - eps) ** 0.6e0
    ff1, ff2, d1, d2 = _plascar_bpol(A, eps, kappa, delta)

    e1 = (2.0 * kappa) / (d1 * (1.0 + delta))
    e2 = (2.0 * kappa) / (d2 * (1.0 - delta))

    plasma_current_ma = (
        a
        * B0
        / qbar
        * 5.0
        * kappa
        / (2.0 * np.pi**2)
        * (np.arcsin(e1) / e1 + np.arcsin(e2) / e2)
        * (ff1 + ff2)
    )
    return 1.0e6 * plasma_current_ma


@relation(
    name="Surface-averaged poloidal field Peng TART",
    tags=("plasma", "spherical_tokamak", "process"),
    outputs="B_p",
)
def poloidal_field_peng(
    q95: float, A: float, eps: float, B0: float, kappa: float, delta: float
) -> float:
    """Surface-averaged poloidal field from the Peng/STAR scaling for tight
    aspect-ratio tokamaks (the TART branch deferred from the plasma_fields
    port).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - Peng, Galambos & Shipe (1992), Fusion Technology 21(3P2A), 1729-1738
    """
    # CHECK
    ff1, ff2, _d1, _d2 = _plascar_bpol(A, eps, kappa, delta)
    qbar = q95 * 1.3e0 * (1.0e0 - eps) ** 0.6e0
    return B0 * (ff1 + ff2) / (2.0 * np.pi * qbar)
