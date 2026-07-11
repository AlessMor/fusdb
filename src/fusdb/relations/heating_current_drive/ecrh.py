"""Electron-cyclotron current-drive efficiency scalings (PROCESS current_drive.py).

Ported from PROCESS ``ElectronCyclotron.electron_cyclotron_fenstermacher`` and
``electron_cyclotron_freethy``. Both return the absolute current-drive
efficiency (A/W) and are gated onto ``eta_cd``. PROCESS's ``culecd`` orchestrator
and the Legendre-based ``eccdef``/``legend`` model are not ported.
"""

import numpy as np

from fusdb.relation import relation
from fusdb.registry import ELECTRON_CHARGE_C, ELECTRON_MASS_KG, EPSILON0

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Current drive efficiency EC Fenstermacher", tags=_TAGS, outputs="eta_cd")
def electron_cyclotron_fenstermacher(
    temp_plasma_electron_density_weighted: float, R: float, n_e_avg: float, dlamee: float
) -> float:
    """Electron-cyclotron heating efficiency (A/W), Fenstermacher model.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    dene20 = n_e_avg / 1e20
    return (0.21e0 * temp_plasma_electron_density_weighted) / (R * dene20 * dlamee)


@relation(name="Current drive efficiency EC Freethy", tags=_TAGS, outputs="eta_cd")
def electron_cyclotron_freethy(
    T_e_avg: float, Z_eff: float, R: float, n_e_avg: float, B0: float,
    n_ecrh_harmonic: float, i_ecrh_wave_mode: float,
) -> float:
    """Electron-cyclotron current-drive efficiency (A/W), Freethy model, with a
    plasma cut-off coupling factor (O-mode: i_ecrh_wave_mode=0, X-mode=1).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS raises on an invalid wave mode; that guard is dropped here.
    """
    # CHECK
    fc = 1 / (2 * np.pi) * ELECTRON_CHARGE_C * B0 / ELECTRON_MASS_KG
    fp = 1 / (2 * np.pi) * np.sqrt(n_e_avg * ELECTRON_CHARGE_C**2 / (ELECTRON_MASS_KG * EPSILON0))
    xi_CD = 0.18e0
    xi_CD *= 4.8e0 / (2 + Z_eff)
    eta_cd = xi_CD * T_e_avg / (3.27e0 * R * (n_e_avg / 1.0e19))
    if i_ecrh_wave_mode == 0:  # O-mode
        f_cutoff = fp
    else:  # X-mode
        f_cutoff = 0.5 * (fc + np.sqrt(n_ecrh_harmonic * fc**2 + 4 * fp**2))
    a = 0.1
    cutoff_factor = 0.5 * (1 + np.tanh((2 / a) * ((n_ecrh_harmonic * fc - f_cutoff) / fp - a)))
    return eta_cd * cutoff_factor


def _legend(zlocal, arg):
    """Legendre conical function P_alpha(arg) and its derivative, order
    alpha = -1/2 + i*sqrt(xisq), via the Abramowitz & Stegun series (PROCESS
    ``legend``).

    # CHECK / TODO: PROCESS c0119fe5 runs this series from ``n = 0``, where the
    # ``/(2n)**2`` term divides by zero -- the function is non-functional there
    # (it raises "not converged" / produces nan). The n=0 term *is* the initial
    # ``palpha = 1.0``, so the series proper starts at n=1; this port fixes that
    # evident off-by-one so the relation computes at all. This is a translation
    # fix, not a physics change, but it means eccdef cannot be verified against
    # PROCESS's (broken) output at the pinned commit -- verified instead by
    # convergence + Abramowitz-Stegun consistency.
    """
    arg2 = min(arg, (1.0e0 - 1.0e-10))
    sinsq = 0.5e0 * (1.0e0 - arg2)
    xisq = 0.25e0 * (32.0e0 * zlocal / (zlocal + 1.0e0) - 1.0e0)
    palpha = pold = pterm = 1.0e0
    palphap = poldp = 0.0e0
    for n in range(1, 10000):
        if (n > 1) and ((n % 20) == 1):
            term1 = 1.0e-10 * max(abs(pold), abs(palpha))
            term2 = 1.0e-10 * max(abs(poldp), abs(palphap))
            if (abs(pold - palpha) < term1) and (abs(poldp - palphap) < term2):
                return palpha, palphap
            pold = palpha
            poldp = palphap
        pterm = pterm * (4.0e0 * xisq + (2.0e0 * n - 1.0e0) ** 2) / (2.0e0 * n) ** 2 * sinsq
        palpha += pterm
        palphap -= n * pterm / (1.0e0 - arg2)
    return palpha, palphap


@relation(name="Current drive efficiency EC Cohen-Legendre", tags=_TAGS, outputs="eta_cd")
def eccdef(
    T_e_avg: float, eps: float, Z_eff: float, cosang: float, dlamie: float,
    n_e_avg: float, R: float,
) -> float:
    """Electron-cyclotron current-drive efficiency (A/W), Cohen/IPDG89 model via
    the Legendre conical function.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS ``eccdef`` returns the normalised efficiency (n I R / P); this port
    applies the ``culecd`` normalisation (divide by dene20 * R) to return the
    absolute A/W efficiency, consistent with the other ``eta_cd`` producers.
    ``cosang`` is the cosine of the poloidal ECCD angle; ``dlamie`` the local
    ion-electron Coulomb logarithm. The negative-efficiency guard is dropped.
    """
    # CHECK
    mcsq = ELECTRON_MASS_KG * 2.9979e8**2 / (1.0e3 * ELECTRON_CHARGE_C)  # keV
    f = 16.0e0 * (T_e_avg / mcsq) ** 2
    fp = 16.0e0 * T_e_avg / mcsq
    lam = 1.0e0
    palpha, palphap = _legend(Z_eff, lam)
    lams = np.sqrt(2.0e0 * eps / (1.0e0 + eps))
    palphas, _ = _legend(Z_eff, lams)
    h = -4.0e0 * lam / (Z_eff + 5.0e0) * (1.0e0 - lams * palpha / (lam * palphas))
    hp = -4.0e0 / (Z_eff + 5.0e0) * (1.0e0 - lams * palphap / palphas)
    facm = 1.5e0
    y = mcsq / (2.0e0 * T_e_avg) * (1.0e0 + eps * cosang)
    ecgam = (
        -7.8e0 * facm * np.sqrt((1.0e0 + eps) / (1.0e0 - eps)) / dlamie
        * (h * fp - 0.5e0 * y * f * hp)
    )
    # culecd normalisation: absolute eta_cd = ecgam / (dene20 * R)
    dene20 = n_e_avg / 1e20
    return ecgam / (dene20 * R)
