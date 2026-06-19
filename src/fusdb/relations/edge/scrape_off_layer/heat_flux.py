"""Scrape-off-layer heat-flux relations."""

import numpy as np

_MU_0 = 1.25663706212e-6

def calc_B_pol_omp(plasma_current, minor_radius):
    """cfspopcon: poloidal field at the outboard midplane."""
    return _MU_0 * plasma_current / (2.0 * np.pi * minor_radius)


def calc_B_tor_omp(magnetic_field_on_axis, major_radius, minor_radius):
    """cfspopcon: toroidal field at the outboard midplane."""
    return magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))


def calc_fieldline_pitch_at_omp(B_t_out_mid, B_pol_out_mid):
    """cfspopcon: B_total / B_poloidal at the outboard midplane."""
    return np.sqrt(B_t_out_mid**2 + B_pol_out_mid**2) / B_pol_out_mid


def calc_parallel_heat_flux_density(
    power_crossing_separatrix, fraction_of_P_SOL_to_divertor, major_radius, minor_radius, lambda_q, fieldline_pitch_at_omp
):
    """cfspopcon: parallel heat flux density entering the flux tube at the outboard midplane."""
    upstream_major_radius = major_radius + minor_radius
    return (
        power_crossing_separatrix
        * fraction_of_P_SOL_to_divertor
        / (2.0 * np.pi * upstream_major_radius * lambda_q)
        * fieldline_pitch_at_omp
    )


def calc_q_perp(power_crossing_separatrix, major_radius, minor_radius, lambda_q):
    """cfspopcon: perpendicular heat flux at the outboard midplane."""
    return power_crossing_separatrix / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q)


def calc_PB_over_R(power_crossing_separatrix, magnetic_field_on_axis, major_radius):
    """cfspopcon: P_sep * B0 / R0 (scales like parallel heat flux entering the SOL)."""
    return power_crossing_separatrix * magnetic_field_on_axis / major_radius


def calc_PBpRnSq(power_crossing_separatrix, magnetic_field_on_axis, q_star, major_radius, average_electron_density):
    """cfspopcon: P_sep * B_pol / (R * n^2) (scales like impurity fraction for detachment)."""
    return (power_crossing_separatrix * (magnetic_field_on_axis / q_star) / major_radius) / (average_electron_density**2.0)
