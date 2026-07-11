"""Current-drive efficiency conversions and driven current (PROCESS current_drive.py).

Ported from PROCESS ``CurrentDrive.calculate_normalised_current_drive_efficiency``
and ``calculate_dimensionless_current_drive_efficiency``, plus the driven-current
relation from the ``current_drive`` orchestrator. The absolute efficiency
``eta_cd`` [A/W] comes from the method-specific scalings (nbi/ecrh/icrf/ebw/lhcd).
"""

from fusdb.relation import relation
from fusdb.registry import ELECTRON_CHARGE_C, EPSILON0, KEV_TO_J

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Normalised current drive efficiency", tags=_TAGS, outputs="eta_cd_norm")
def calculate_normalised_current_drive_efficiency(eta_cd: float, n_e_avg: float, R: float) -> float:
    """Normalised current-drive efficiency gamma [10^20 A/Wm^2].

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return eta_cd * (n_e_avg * R) * 1.0e-20


@relation(name="Auxiliary driven current", tags=_TAGS, outputs="c_hcd_driven")
def auxiliary_driven_current(eta_cd: float, p_hcd_injected: float) -> float:
    """Current driven by an auxiliary heating system [A] = eta_cd * P.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return eta_cd * p_hcd_injected


@relation(name="Dimensionless current drive efficiency", tags=_TAGS, outputs="eta_cd_dimensionless")
def calculate_dimensionless_current_drive_efficiency(
    n_e_avg: float, R: float, T_e_avg: float, c_hcd_driven: float, p_hcd_injected: float
) -> float:
    """Dimensionless current-drive efficiency zeta (Poli/Luce).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        (ELECTRON_CHARGE_C**3 / EPSILON0**2)
        * ((n_e_avg * R) / (T_e_avg * KEV_TO_J))
        * (c_hcd_driven / p_hcd_injected)
    )
