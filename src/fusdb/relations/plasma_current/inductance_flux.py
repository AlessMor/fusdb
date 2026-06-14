"""Inductance and flux-consumption relations (UNDECORATED cfspopcon scaffolds).

Source: cfspopcon/formulas/plasma_current/flux_consumption/{flux_consumption.py,inductances.py}

These compute the internal/external/resistive/poloidal-field flux, the flux needed
from the central solenoid over ramp-up, the maximum flattop duration and the breakdown
flux consumption (Barr 2018 power-balance model). They are NOT yet fusdb @relation's:
review the formula + variable/unit mapping (plasma_current->I_p in A, major_radius->R,
loop_voltage in V, fluxes in Wb), then add @relation to activate.

The external_inductance / vertical_field_mutual_inductance / invmu_0_dLedR /
vertical_magnetic_field functions additionally need cfspopcon's analytical helper
functions (calc_fa..calc_fh in inductance_analytical_functions.py) and the
SurfaceInductanceCoeffs coefficient sets; port those before activating, see
cfspopcon/formulas/plasma_current/flux_consumption/inductances.py.
"""

import numpy as np

_MU_0 = 1.25663706212e-6  # T*m/A


# TODO(cfspopcon): activate as a fusdb relation (internal_inductivity output).
def calc_internal_inductivity(cylindrical_safety_factor, safety_factor_on_axis=1.0):
    """cfspopcon: normalized internal inductance, circular cross-section (Wesson pg.120)."""
    return np.log(1.65 + 0.89 * ((cylindrical_safety_factor / safety_factor_on_axis) - 1.0))


# TODO(cfspopcon): activate as a fusdb relation (internal_inductance output, H).
def calc_internal_inductance_for_cylindrical(major_radius, internal_inductivity):
    """cfspopcon: internal inductance, circular cross-section (Barr 2018)."""
    return _MU_0 * major_radius * internal_inductivity / 2.0


# TODO(cfspopcon): activate as a fusdb relation (internal_flux output, Wb).
def calc_internal_flux(plasma_current, internal_inductance):
    """cfspopcon: internal_flux = plasma_current * internal_inductance (Barr 2018)."""
    return plasma_current * internal_inductance


# TODO(cfspopcon): activate as a fusdb relation (external_flux output, Wb).
def calc_external_flux(plasma_current, external_inductance):
    """cfspopcon: external_flux = plasma_current * external_inductance (Barr 2018)."""
    return plasma_current * external_inductance


# TODO(cfspopcon): activate as a fusdb relation (resistive_flux output, Wb).
def calc_resistive_flux(plasma_current, major_radius, ejima_coefficient):
    """cfspopcon: resistive_flux = ejima_coefficient * mu_0 * I_p * R (Gribov 2007)."""
    return ejima_coefficient * _MU_0 * plasma_current * major_radius


# TODO(cfspopcon): activate as a fusdb relation (poloidal_field_flux output, Wb).
def calc_poloidal_field_flux(vertical_field_mutual_inductance, vertical_magnetic_field, major_radius):
    """cfspopcon: surface flux from the vertical field for radial force balance (Barr 2018)."""
    return np.pi * major_radius**2 * vertical_field_mutual_inductance * vertical_magnetic_field


# TODO(cfspopcon): activate as a fusdb relation (flux_needed_from_CS_over_rampup output, Wb).
def calc_flux_needed_from_solenoid_over_rampup(internal_flux, external_flux, resistive_flux, poloidal_field_flux):
    """cfspopcon: total CS flux needed over ramp-up (PF-coil contribution subtracted)."""
    return internal_flux + external_flux + resistive_flux - poloidal_field_flux


# TODO(cfspopcon): activate as a fusdb relation (max_flattop_duration output, s).
def calc_max_flattop_duration(total_flux_available_from_CS, flux_needed_from_CS_over_rampup, loop_voltage):
    """cfspopcon: maximum flattop duration drivable by the central solenoid."""
    max_flux_for_flattop = total_flux_available_from_CS - flux_needed_from_CS_over_rampup
    return max_flux_for_flattop / loop_voltage


# TODO(cfspopcon): activate as a fusdb relation (breakdown_flux_consumption output, Wb).
def calc_breakdown_flux_consumption(major_radius):
    """cfspopcon: resistive flux required for breakdown (Sugihara)."""
    return 0.073 * major_radius - 0.00665


# TODO(cfspopcon): port external_inductance / vertical_field_mutual_inductance /
#   calc_invmu_0_dLedR / calc_vertical_magnetic_field from inductances.py — they need
#   the calc_fa..calc_fh analytical helpers and the SurfaceInductanceCoeffs coefficient
#   sets (Barr/Hirshman). Left as a pointer to keep this scaffold self-contained.
