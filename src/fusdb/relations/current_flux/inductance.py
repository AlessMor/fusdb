"""Plasma inductance relations.

Internal inductivity (normalised internal inductance) and the internal
self-inductance, for circular (cylindrical) and non-cylindrical cross-sections.
The external/vertical Barr surface-inductance sub-model lives in
``inductance_surface.py``.
"""

from typing import Any

import numpy as np

from fusdb import relation
from fusdb.registry import MU0


@relation(
    name="Internal inductivity",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="internal_inductivity",
)
def calc_internal_inductivity(q_cyl: Any, safety_factor_on_axis: Any = 1.0) -> Any:
    """Normalised internal inductance for an assumed circular plasma cross-section.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    Tokamaks (pg.120) :cite:`wesson_tokamaks_2011`.

    Args:
        q_cyl: [~] :term:`glossary link<cylindrical_safety_factor>`
        safety_factor_on_axis: [~] :term:`glossary link<safety_factor_on_axis>`

    Returns:
        internal_inductivity [~]
    """
    # CHECK
    return np.log(1.65 + 0.89 * ((q_cyl / safety_factor_on_axis) - 1.0))


@relation(
    name="Internal inductance (cylindrical)",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="internal_inductance",
)
def calc_internal_inductance_for_cylindrical(R: Any, internal_inductivity: Any) -> Any:
    """Internal inductance of the plasma (circular cross-section).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    :cite:`Barr_2018`.
    """
    # CHECK
    return MU0 * R * internal_inductivity / 2.0


@relation(
    name="Internal inductance (non-cylindrical)",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="internal_inductance",
)
def calc_internal_inductance_for_noncylindrical(V_p: Any, L_p: Any, internal_inductivity: Any) -> Any:
    """Internal inductance of the plasma (general cross-section).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    :cite:`Barr_2018`. ``L_p`` is the poloidal circumference.
    """
    # CHECK
    return MU0 * internal_inductivity * V_p / (L_p**2)
