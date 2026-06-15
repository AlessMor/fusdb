# TODO(low): add from cfspopcon
    #2PM
    # ...
#
# cfspopcon ports (UNDECORATED scaffolds), source:
# cfspopcon/formulas/scrape_off_layer/{heat_flux_density.py,lambda_q.py,separatrix_density.py}
# and cfspopcon/formulas/metrics/heat_exhaust_metrics.py
# Review formula + variable/unit mapping (power_crossing_separatrix->P_sep in W,
# magnetic_field_on_axis->B0, major/minor_radius->R/a, plasma_current->I_p,
# average_total_pressure->p_th, average_electron_density->n_avg). lambda_q is returned
# in mm and q_parallel in GW/m^2 in cfspopcon. Then add @relation to activate.
#
# The full two-point-model detachment solve (two_point_model_fixed_tet,
# calc_edge_impurity_concentration, calc_impurity_charge_state) lives in
# cfspopcon/formulas/scrape_off_layer/two_point_model/ and is not ported here yet.

import numpy as np

_MU_0 = 1.25663706212e-6  # T*m/A


# TODO(cfspopcon): activate as a fusdb relation (B_pol_out_mid output, T). plasma_current in A.
def calc_B_pol_omp(plasma_current, minor_radius):
    """cfspopcon: poloidal field at the outboard midplane."""
    return _MU_0 * plasma_current / (2.0 * np.pi * minor_radius)


# TODO(cfspopcon): activate as a fusdb relation (B_t_out_mid output, T).
def calc_B_tor_omp(magnetic_field_on_axis, major_radius, minor_radius):
    """cfspopcon: toroidal field at the outboard midplane."""
    return magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))


# TODO(cfspopcon): activate as a fusdb relation (fieldline_pitch_at_omp output).
def calc_fieldline_pitch_at_omp(B_t_out_mid, B_pol_out_mid):
    """cfspopcon: B_total / B_poloidal at the outboard midplane."""
    return np.sqrt(B_t_out_mid**2 + B_pol_out_mid**2) / B_pol_out_mid


# TODO(cfspopcon): activate as a fusdb relation (lambda_q output, mm). SPARC uses EichRegression15.
def calc_lambda_q_with_eich_regression_15(
    power_crossing_separatrix, major_radius, B_pol_out_mid, inverse_aspect_ratio, lambda_q_factor=1.0
):
    """cfspopcon: lambda_q from Eich regression #15 (Eich 2013, Table 3)."""
    lambda_q = 1.35 * major_radius**0.04 * B_pol_out_mid**-0.92 * inverse_aspect_ratio**0.42
    if power_crossing_separatrix > 0:
        return lambda_q_factor * lambda_q * power_crossing_separatrix**-0.02
    return lambda_q_factor * lambda_q


# TODO(cfspopcon): activate as a fusdb relation (q_parallel output, GW/m^2).
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


# TODO(cfspopcon): activate as a fusdb relation (q_perp output, MW/m^2).
def calc_q_perp(power_crossing_separatrix, major_radius, minor_radius, lambda_q):
    """cfspopcon: perpendicular heat flux at the outboard midplane."""
    return power_crossing_separatrix / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q)


# TODO(cfspopcon): activate as a fusdb relation (separatrix_electron_density output).
def calc_separatrix_electron_density(nesep_over_nebar, average_electron_density):
    """cfspopcon: separatrix electron density = nesep_over_nebar * average_electron_density."""
    return nesep_over_nebar * average_electron_density


# TODO(cfspopcon): activate as a fusdb relation (PB_over_R output).
def calc_PB_over_R(power_crossing_separatrix, magnetic_field_on_axis, major_radius):
    """cfspopcon: P_sep * B0 / R0 (scales like parallel heat flux entering the SOL)."""
    return power_crossing_separatrix * magnetic_field_on_axis / major_radius


# TODO(cfspopcon): activate as a fusdb relation (PBpRnSq output).
def calc_PBpRnSq(power_crossing_separatrix, magnetic_field_on_axis, q_star, major_radius, average_electron_density):
    """cfspopcon: P_sep * B_pol / (R * n^2) (scales like impurity fraction for detachment)."""
    return (power_crossing_separatrix * (magnetic_field_on_axis / q_star) / major_radius) / (average_electron_density**2.0)
