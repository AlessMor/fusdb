"""Plasma diamagnetic current-fraction scalings.

Ported from PROCESS ``process/models/physics/plasma_current.py``
(``PlasmaDiamagneticCurrent``). The ``i_diamagnetic_current`` enum dispatcher is
split into one relation per scaling, gated in variables.yaml.
"""

from fusdb.relation import relation


@relation(
    name="Diamagnetic current fraction Hender",
    tags=("plasma", "current_drive", "spherical_tokamak", "process"),
    outputs="f_diamagnetic",
)
def diamagnetic_fraction_hender(beta: float) -> float:
    """Diamagnetic current fraction from the Hender tight-aspect-ratio fit.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return beta / 2.8


@relation(
    name="Diamagnetic current fraction SCENE",
    tags=("plasma", "current_drive", "tokamak", "process"),
    outputs="f_diamagnetic",
)
def diamagnetic_fraction_scene(beta: float, q95: float, q0: float) -> float:
    """Diamagnetic current fraction from the SCENE fit (Tim Hender).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return beta * (0.1 * q95 / q0 + 0.44) * 0.414
