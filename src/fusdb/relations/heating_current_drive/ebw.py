"""Electron-Bernstein-wave current-drive efficiency scaling (PROCESS current_drive.py).

Ported from PROCESS ``ElectronBernstein.electron_bernstein_freethy``. Returns the
absolute current-drive efficiency (A/W), gated onto ``eta_cd``.
"""

import numpy as np

from fusdb import relation
from fusdb.registry import ELECTRON_CHARGE_C, ELECTRON_MASS_KG, EPSILON0

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Current drive efficiency EBW Freethy", tags=_TAGS, outputs="eta_cd")
def electron_bernstein_freethy(
    T_e_avg: float, R: float, n_e_avg: float, B0: float, n_ecrh_harmonic: float, xi_ebw: float
) -> float:
    """Electron-Bernstein-wave current-drive efficiency (A/W), Freethy model,
    with a density cut-off coupling factor.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    dene20 = n_e_avg / 1e20
    eta_cd_norm = (xi_ebw / 32.7e0) * T_e_avg
    eta_cd = eta_cd_norm / (dene20 * R)
    a = 0.1e0
    fc = 1.0e0 / (2.0e0 * np.pi) * n_ecrh_harmonic * ELECTRON_CHARGE_C * B0 / ELECTRON_MASS_KG
    fp = 1.0e0 / (2.0e0 * np.pi) * np.sqrt(
        dene20 * 1.0e20 * ELECTRON_CHARGE_C**2 / (ELECTRON_MASS_KG * EPSILON0)
    )
    density_factor = 0.5e0 * (1.0e0 + np.tanh((2.0e0 / a) * ((fp - fc) / fp - a)))
    return eta_cd * density_factor
