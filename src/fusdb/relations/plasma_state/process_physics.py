"""Standalone physics formulas cherry-picked from PROCESS physics.py.

Only self-contained formulas that produce fusdb quantities are ported; the
Physics/PlasmaBeta/PlasmaInductance/DetailedPhysics class orchestrators,
composition/ohmic plumbing, I/O, and the generic (mass/charge-parameterised)
plasma-diagnostic utilities are not.
"""

import numpy as np

from fusdb import relation
from fusdb.registry import ELECTRON_CHARGE_C, EPSILON0, KEV_TO_J


# NOTE: PROCESS's Wesson current-profile index (alphaj = qstar/q0 - 1) is
# intentionally NOT ported as a relation. Deriving the defaulted ``alphaj``
# from qstar/q0 over-determines the pinned-reconcile (popcon) solve, exactly as
# the ``alphap = alphan + alphat`` relation did (see profiles/central_values.py).
# ``alphaj`` stays a pure input with its default; its consumers (Connor-Hastie,
# Hoang, Wilson, central current density) take it as an input.


@relation(
    name="Pfirsch-Schluter current fraction (SCENE)",
    tags=("plasma", "current_drive", "tokamak", "process"),
    outputs="f_pfirsch_schluter",
)
def ps_fraction_scene(beta: float) -> float:
    """Pfirsch-Schluter current fraction, SCENE fit (Hender 2019): -0.09 * beta.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return -9e-2 * beta


@relation(
    name="Ion-electron equilibration power density",
    tags=("plasma", "process"),
    outputs="P_ie_equilibration",
)
def rether(
    alphan: float, alphat: float, n_e_avg: float, dlamie: float, T_e_avg: float,
    T_i_avg: float, n_charge_plasma_effective_mass_weighted_vol_avg: float,
) -> float:
    """Ion-electron equilibration power density [W/m^3].

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS returns MW/m^3; converted to W/m^3 here. ``dlamie`` is the
    ion-electron Coulomb logarithm.
    """
    # CHECK
    profie = (1.0 + alphan) ** 2 / ((2.0 * alphan - 0.5 * alphat + 1.0) * np.sqrt(1.0 + alphat))
    conie = (
        2.42165e-41 * dlamie * n_e_avg**2
        * n_charge_plasma_effective_mass_weighted_vol_avg * profie
    )
    return 1.0e6 * (conie * (T_i_avg - T_e_avg) / (T_e_avg**1.5))


@relation(
    name="Debye length",
    tags=("plasma", "process"),
    outputs="debye_length",
)
def calculate_debye_length(T_e_avg: float, n_e_avg: float) -> float:
    """Electron Debye length [m].

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return ((EPSILON0 * T_e_avg * KEV_TO_J) / (n_e_avg * ELECTRON_CHARGE_C**2)) ** 0.5
