"""Flux-consumption relations."""

import numpy as np

_MU_0 = 1.25663706212e-6

def calc_internal_flux(plasma_current, internal_inductance):
    """cfspopcon: internal_flux = plasma_current * internal_inductance (Barr 2018)."""
    return plasma_current * internal_inductance


def calc_external_flux(plasma_current, external_inductance):
    """cfspopcon: external_flux = plasma_current * external_inductance (Barr 2018)."""
    return plasma_current * external_inductance


def calc_resistive_flux(plasma_current, major_radius, ejima_coefficient):
    """cfspopcon: resistive_flux = ejima_coefficient * mu_0 * I_p * R (Gribov 2007)."""
    return ejima_coefficient * _MU_0 * plasma_current * major_radius


def calc_poloidal_field_flux(vertical_field_mutual_inductance, vertical_magnetic_field, major_radius):
    """cfspopcon: surface flux from the vertical field for radial force balance (Barr 2018)."""
    return np.pi * major_radius**2 * vertical_field_mutual_inductance * vertical_magnetic_field


def calc_flux_needed_from_solenoid_over_rampup(internal_flux, external_flux, resistive_flux, poloidal_field_flux):
    """cfspopcon: total CS flux needed over ramp-up (PF-coil contribution subtracted)."""
    return internal_flux + external_flux + resistive_flux - poloidal_field_flux


def calc_max_flattop_duration(total_flux_available_from_CS, flux_needed_from_CS_over_rampup, loop_voltage):
    """cfspopcon: maximum flattop duration drivable by the central solenoid."""
    max_flux_for_flattop = total_flux_available_from_CS - flux_needed_from_CS_over_rampup
    return max_flux_for_flattop / loop_voltage


def calc_breakdown_flux_consumption(major_radius):
    """cfspopcon: resistive flux required for breakdown (Sugihara)."""
    return 0.073 * major_radius - 0.00665
