"""Lower-hybrid current-drive efficiency scalings (PROCESS current_drive.py).

Ported from PROCESS ``LowerHybrid.lower_hybrid_fenstermacher`` and
``lower_hybrid_ehst``. Both return the absolute current-drive efficiency (A/W)
and are gated onto ``eta_cd``. PROCESS's ``cullhy``/``lhrad``/``lheval``
orchestration is not ported.
"""

from fusdb import relation

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Current drive efficiency LH Fenstermacher", tags=_TAGS, outputs="eta_cd")
def lower_hybrid_fenstermacher(T_e_avg: float, R: float, n_e_avg: float) -> float:
    """Lower-hybrid current-drive efficiency (A/W), Fenstermacher model.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    dene20 = n_e_avg / 1e20
    return (0.36e0 * (1.0e0 + (T_e_avg / 25.0e0) ** 1.16e0)) / (R * dene20)


@relation(name="Current drive efficiency LH Ehst", tags=_TAGS, outputs="eta_cd")
def lower_hybrid_ehst(T_e_avg: float, beta: float, R: float, n_e_avg: float, Z_eff: float) -> float:
    """Lower-hybrid current-drive efficiency (A/W), Ehst-Karney model.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    dene20 = n_e_avg / 1e20
    return (
        ((T_e_avg**0.77 * (0.034 + 0.196 * beta)) / (R * dene20))
        * (
            32.0 / (5.0 + Z_eff)
            + 2.0
            + (12.0 * (6.0 + Z_eff)) / (5.0 + Z_eff) / (3.0 + Z_eff)
            + 3.76 / Z_eff
        )
        / 12.507
    )
