"""Ion-cyclotron current-drive efficiency scaling (PROCESS current_drive.py).

Ported from PROCESS ``IonCyclotron.ion_cyclotron_ipdg89``. Returns the absolute
current-drive efficiency (A/W), gated onto ``eta_cd``.
"""

from fusdb import relation

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Current drive efficiency IC IPDG89", tags=_TAGS, outputs="eta_cd")
def ion_cyclotron_ipdg89(
    temp_plasma_electron_density_weighted: float, Z_eff: float, R: float, n_e_avg: float
) -> float:
    """Ion-cyclotron heating efficiency (A/W), IPDG89 model.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    The 0.1 factor expresses the temperature in 10 keV units; the density and
    major-radius terms restore the absolute (not normalised) efficiency.
    """
    # CHECK
    dene20 = n_e_avg / 1e20
    return ((0.63e0 * 0.1e0 * temp_plasma_electron_density_weighted) / (2.0e0 + Z_eff)) / (
        R * dene20
    )
