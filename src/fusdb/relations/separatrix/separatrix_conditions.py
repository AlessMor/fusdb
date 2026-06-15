# TODO(low): add from cfspopcon (once profiles are used)
    # separatrix operational space
#
# cfspopcon ports (UNDECORATED scaffolds), source:
# cfspopcon/formulas/separatrix_conditions/{power_crossing_separatrix.py,threshold_power.py}
# Review formula + variable/unit mapping (P_in->P_loss, P_radiation->P_rad,
# power_crossing_separatrix->P_sep, P_LH_thresh->P_LH, average_electron_density->n_avg
# in 1e19 m^-3, plasma_current->I_p in MA, average_ion_mass->afuel, surface_area->A_p),
# then add @relation to activate. NOTE: fusdb already has a 'L-H transition threshold
# power' relation with a different (simpler) scaling — compare before activating this.

import numpy as np


# TODO(cfspopcon): activate as a fusdb relation (power_crossing_separatrix output).
def calc_power_crossing_separatrix(P_in, P_radiation):
    """cfspopcon: P_sep = max(P_in - P_radiation, 0)."""
    return np.maximum(P_in - P_radiation, 0.0)


# TODO(cfspopcon): activate as a fusdb relation (P_LH_thresh output, MW).
#   Martin-2008 scaling with mass correction + Ryter-2014 low-density branch.
def calc_LH_transition_threshold_power(
    plasma_current,
    magnetic_field_on_axis,
    minor_radius,
    major_radius,
    surface_area,
    average_ion_mass,
    average_electron_density,
    confinement_threshold_scalar=1.0,
):
    """cfspopcon: L-H transition threshold power (Martin 2008 + Ryter 2014 low-n branch).

    plasma_current [MA], average_electron_density [1e19 m^-3].
    """

    def _calc_Martin_LH_threshold(electron_density):
        _DEUTERIUM_MASS_NUMBER = 2.0
        return (
            0.0488 * ((electron_density / 10.0) ** 0.717) * (magnetic_field_on_axis**0.803) * (surface_area**0.941)
        ) * (_DEUTERIUM_MASS_NUMBER / average_ion_mass)

    # Ryter 2014, equation 3 (low-density rollover)
    neMin19 = (
        0.7 * (plasma_current**0.34) * (magnetic_field_on_axis**0.62) * (minor_radius**-0.95) * ((major_radius / minor_radius) ** 0.4)
    )

    if average_electron_density < neMin19:
        P_LH_thresh = _calc_Martin_LH_threshold(electron_density=neMin19)
        return (P_LH_thresh * (neMin19 / average_electron_density) ** 2.0) * confinement_threshold_scalar
    P_LH_thresh = _calc_Martin_LH_threshold(electron_density=average_electron_density)
    return P_LH_thresh * confinement_threshold_scalar


# TODO(cfspopcon): activate as a fusdb relation (ratio_of_P_SOL_to_P_LH output).
def calc_ratio_P_LH(power_crossing_separatrix, P_LH_thresh):
    """cfspopcon: ratio_of_P_SOL_to_P_LH = power_crossing_separatrix / P_LH_thresh."""
    return power_crossing_separatrix / P_LH_thresh
