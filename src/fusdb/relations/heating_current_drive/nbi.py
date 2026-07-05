"""Neutral-beam current-drive efficiency scalings (PROCESS current_drive.py).

Ported from PROCESS ``NeutralBeam.etanb2``/``etanb``. Both return the absolute
current-drive efficiency (A/W) and are gated onto ``eta_cd`` in variables.yaml.
PROCESS's beam-geometry/shine-through orchestration (``iternb``/``culnbi``/
``sigbeam``/``xlmbdabi``/``cfnbi``) is not ported.
"""

import numpy as np

from fusdb import relation
from fusdb.registry import ELECTRON_CHARGE_C, ELECTRON_MASS_KG, PROTON_MASS_KG

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Current drive efficiency NBI ITER-1990", tags=_TAGS, outputs="eta_cd")
def etanb2(
    m_beam_amu: float, alphan: float, alphat: float, A: float, n_e_avg: float, n_la: float,
    e_beam_kev: float, f_radius_beam_tangency_rmajor: float, fshine: float, R: float, a: float,
    temp_plasma_electron_density_weighted: float, Z_eff: float,
) -> float:
    """Neutral-beam current-drive efficiency (A/W), ITER-1990 formulation with
    the AEA FUS 172 correction terms.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS raises when the beam tangency radius exceeds the plasma (beam misses
    the plasma); that guard is dropped here (a non-finite result propagates).
    """
    # CHECK
    zbeam = 1.0
    bbd = 1.0
    dene20 = n_e_avg / 1e20
    dnla20 = n_la / 1e20
    ecrit = 0.01 * m_beam_amu * temp_plasma_electron_density_weighted
    ebmev = e_beam_kev / 1e3
    xjs = ebmev / (bbd * ecrit)
    xj = np.sqrt(xjs)
    yj = 0.8 * Z_eff / m_beam_amu
    j0 = xjs / (4.0 + 3.0 * yj + xjs * (xj + 1.39 + 0.61 * yj**0.7))
    epseff = min(0.2, (0.5 / A))
    gfac = (1.55 + 0.85 / Z_eff) * np.sqrt(epseff) - (0.2 + 1.55 / Z_eff) * epseff
    ffac = 1.0 - (zbeam / Z_eff) * (1.0 - gfac)
    nnorm = 1.0
    r = max(R, R * f_radius_beam_tangency_rmajor)
    eps1 = a / r
    d = R * np.sqrt((1.0 + eps1) ** 2 - f_radius_beam_tangency_rmajor**2)
    epsitr = 2.15 / 6.0
    dnorm = 6.0 * np.sqrt(2.0 * epsitr + epsitr**2)
    ebnorm = ebmev * ((nnorm * dnorm) / (dnla20 * d)) ** (1.0 / 0.78)
    abd = (
        0.107
        * (1.0 - 0.35 * alphan + 0.14 * alphan**2)
        * (1.0 - 0.21 * alphat)
        * (1.0 - 0.2 * ebnorm + 0.09 * ebnorm**2)
    )
    gamnb = (
        5.0 * abd * 0.1 * temp_plasma_electron_density_weighted * (1.0 - fshine)
        * f_radius_beam_tangency_rmajor * j0 / 0.2 * ffac
    )
    return gamnb / (dene20 * R)


@relation(name="Current drive efficiency NBI simplified", tags=_TAGS, outputs="eta_cd")
def etanb(
    m_beam_amu: float, alphan: float, alphat: float, A: float, n_e_avg: float,
    e_beam_kev: float, R: float, temp_plasma_electron_density_weighted: float, Z_eff: float,
) -> float:
    """Neutral-beam current-drive efficiency (A/W), simplified ITER-1990
    formulation (no tangency-radius geometry).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    zbeam = 1.0
    bbd = 1.0
    dene20 = 1e-20 * n_e_avg
    xjs = e_beam_kev / (bbd * 10.0 * m_beam_amu * temp_plasma_electron_density_weighted)
    xj = np.sqrt(xjs)
    yj = 0.8 * Z_eff / m_beam_amu
    rjfunc = xjs / (4.0 + 3.0 * yj + xjs * (xj + 1.39 + 0.61 * yj**0.7))
    epseff = 0.5 / A
    gfac = (1.55 + 0.85 / Z_eff) * np.sqrt(epseff) - (0.2 + 1.55 / Z_eff) * epseff
    ffac = 1.0 / zbeam - (1.0 - gfac) / Z_eff
    abd = (
        0.107
        * (1.0 - 0.35 * alphan + 0.14 * alphan**2)
        * (1.0 - 0.21 * alphat)
        * (1.0 - 0.2e-3 * e_beam_kev + 0.09e-6 * e_beam_kev**2)
    )
    return (
        abd * (5.0 / R) * (0.1 * temp_plasma_electron_density_weighted / dene20)
        * rjfunc / 0.2 * ffac
    )


# ── Neutral-beam micro-physics helpers (PROCESS NeutralBeam) ──────────────────
# Standalone formulas used by the (skipped) culnbi/iternb beam-deposition
# orchestrators. The beam-line geometry and power-balance bookkeeping in those
# methods are still not ported; the closed-form beam chord length (dpath) is
# computed here directly from R/eps/tangency, as PROCESS does.

_SIGBEAM_A = np.array([
    [[4.4, -2.49e-2], [7.46e-2, 2.27e-3], [3.16e-3, -2.78e-5]],
    [[2.3e-1, -1.15e-2], [-2.55e-3, -6.2e-4], [1.32e-3, 3.38e-5]],
])
_SIGBEAM_B = np.array([
    [[[-2.36, -1.49, -1.41, -1.03], [0.185, -0.0154, -4.08e-4, 0.106]],
     [[-0.25, -0.119, -0.108, -0.0558], [-0.0381, -0.015, -0.0138, -3.72e-3]]],
    [[[0.849, 0.518, 0.477, 0.322], [-0.0478, 7.18e-3, 1.57e-3, -0.0375]],
     [[0.0677, 0.0292, 0.0259, 0.0124], [0.0105, 3.66e-3, 3.33e-3, 8.61e-4]]],
    [[[-0.0588, -0.0336, -0.0305, -0.0187], [4.34e-3, 3.41e-4, 7.35e-4, 3.53e-3]],
     [[-4.48e-3, -1.79e-3, -1.57e-3, -7.43e-4], [-6.76e-4, -2.04e-4, -1.86e-4, -5.12e-5]]],
])
_SIGBEAM_Z = np.array([2.0, 6.0, 8.0, 26.0])


@relation(name="Beam stopping cross-section", tags=_TAGS, outputs="beam_stopping_cross_section")
def sigbeam(
    e_beam_kev: float, m_beam_amu: float, T_e_avg: float, n_e_avg: float,
    c_He: float = 0.0, c_C: float = 0.0, c_O: float = 0.0, c_Fe: float = 0.0,
) -> float:
    """Stopping cross-section (m^2) for a hydrogen beam in a fusion plasma
    (Janev, Boley & Post 1989).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``eb`` is the beam energy per nucleon (keV/amu); ``c_He``/``c_C``/``c_O``/
    ``c_Fe`` are the He/C/O/Fe densities relative to the electron density.
    """
    # CHECK
    eb = e_beam_kev / m_beam_amu
    nn = np.array([c_He, c_C, c_O, c_Fe])
    nen = n_e_avg * 1e-19
    s1 = 0.0
    for k in range(2):
        for j in range(3):
            for i in range(2):
                s1 += _SIGBEAM_A[i, j, k] * np.log(eb) ** i * np.log(nen) ** j * np.log(T_e_avg) ** k
    sz = 0.0
    for l_ in range(4):
        for k in range(2):
            for j in range(2):
                for i in range(3):
                    sz += (
                        _SIGBEAM_B[i, j, k, l_]
                        * np.log(eb) ** i * np.log(nen) ** j * np.log(T_e_avg) ** k
                        * nn[l_] * _SIGBEAM_Z[l_] * (_SIGBEAM_Z[l_] - 1.0)
                    )
    return max(1e-20 * (np.exp(s1) / eb * (1.0 + sz)), 1e-23)


@relation(name="Beam-ion Coulomb logarithm", tags=_TAGS, outputs="beam_ion_coulomb_log")
def xlmbdabi(
    m_beam_amu: float, afuel: float, e_beam_kev: float,
    temp_plasma_electron_density_weighted: float, n_e_avg: float,
) -> float:
    """Coulomb logarithm for beam-ion (fast-ion / background-ion) collisions
    (Mikkelson & Singer 1983).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``m_beam_amu`` is the fast-ion mass, ``afuel`` the background-ion mass.
    """
    # CHECK
    x1 = (temp_plasma_electron_density_weighted / 10.0) * (e_beam_kev / 1000.0) * m_beam_amu / (n_e_avg / 1e20)
    x2 = afuel / (afuel + m_beam_amu)
    return 23.7 + np.log(x2 * np.sqrt(x1))


@relation(name="Beam fraction coupled to ions", tags=_TAGS, outputs="f_p_beam_injected_ions")
def cfnbi(
    m_beam_amu: float, e_beam_kev: float, temp_plasma_electron_density_weighted: float,
    n_e_avg: float, n_charge_plasma_effective_mass_weighted_vol_avg: float, dlamie: float,
) -> float:
    """Fraction of fast (beam) particle energy coupled to the ions.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``dlamie`` is the ion-electron Coulomb logarithm.
    """
    # CHECK
    atmdt = 2.5
    c = 3.0e8
    me = ELECTRON_MASS_KG
    # Beam-ion Coulomb log against the mean D-T background (PROCESS calls xlmbdabi)
    x1 = (temp_plasma_electron_density_weighted / 10.0) * (e_beam_kev / 1000.0) * m_beam_amu / (n_e_avg / 1e20)
    xlmbdai = 23.7 + np.log((atmdt / (atmdt + m_beam_amu)) * np.sqrt(x1))
    sumln = n_charge_plasma_effective_mass_weighted_vol_avg * xlmbdai / dlamie
    xlnrat = (3.0e0 * np.sqrt(np.pi) / 4.0e0 * me / PROTON_MASS_KG * sumln) ** (2.0e0 / 3.0e0)
    ve = c * np.sqrt(2.0e0 * temp_plasma_electron_density_weighted / 511.0e0)
    ecritfi = (
        m_beam_amu * PROTON_MASS_KG * ve * ve * xlnrat
        / (2.0e0 * ELECTRON_CHARGE_C * 1.0e3)
    )
    x = np.sqrt(e_beam_kev / ecritfi)
    t1 = np.log((x * x - x + 1.0e0) / ((x + 1.0e0) ** 2))
    thx = (2.0e0 * x - 1.0e0) / np.sqrt(3.0e0)
    t2 = 2.0e0 * np.sqrt(3.0e0) * (np.arctan(thx) + np.pi / 6.0e0)
    return (t1 + t2) / (3.0e0 * x * x)


@relation(name="Beam shine-through fraction", tags=_TAGS, outputs="fshine")
def beam_shine_through_fraction(
    R: float, eps: float, f_radius_beam_tangency_rmajor: float, n_la: float,
    beam_stopping_cross_section: float,
) -> float:
    """Neutral-beam shine-through fraction from the beam optical depth
    (PROCESS ``culnbi``): fshine = exp(-2 * dpath * n_line * sigma).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    The beam chord length to the plasma centre is the closed form
    ``dpath = R * sqrt((1 + eps)^2 - f_tangency^2)``. PROCESS raises when the
    beam misses the plasma (tangency > 1 + eps); that guard is dropped here.
    """
    # CHECK
    dpath = R * np.sqrt((1.0e0 + eps) ** 2 - f_radius_beam_tangency_rmajor**2)
    fshine = np.exp(-2.0e0 * dpath * n_la * beam_stopping_cross_section)
    return max(fshine, 1.0e-20)
