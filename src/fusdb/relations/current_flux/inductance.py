"""Plasma inductance relations."""

import numpy as np

_MU_0 = 1.25663706212e-6

def calc_internal_inductivity(cylindrical_safety_factor, safety_factor_on_axis=1.0):
    """cfspopcon: normalized internal inductance, circular cross-section (Wesson pg.120)."""
    return np.log(1.65 + 0.89 * ((cylindrical_safety_factor / safety_factor_on_axis) - 1.0))


def calc_internal_inductance_for_cylindrical(major_radius, internal_inductivity):
    """cfspopcon: internal inductance, circular cross-section (Barr 2018)."""
    return _MU_0 * major_radius * internal_inductivity / 2.0
