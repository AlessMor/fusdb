"""Scrape-off-layer heat-flux relations.

cfspopcon displays q_parallel in GW/m^2 and q_perp in MW/m^2; fusdb stores both
in SI (W/m^2). The formulas are unit-consistent in SI, so no rescaling is needed.
"""

import numpy as np

from fusdb.relation import relation
from fusdb.registry import MU0


@relation(
    name="Poloidal field at outboard midplane",
    tags=("power_exhaust", "tokamak"),
    outputs="B_pol_out_mid",
)
def calc_B_pol_omp(I_p, a):
    """Calculate the poloidal magnetic field at the outboard midplane.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
        I_p: [A] :term:`glossary link<plasma_current>`
        a: [m] :term:`glossary link<minor_radius>`

    Returns:
         B_pol_out_mid [T]
    """
    # CHECK
    return MU0 * I_p / (2.0 * np.pi * a)


@relation(
    name="Toroidal field at outboard midplane",
    tags=("power_exhaust", "tokamak"),
    outputs="B_t_out_mid",
)
def calc_B_tor_omp(B0, R, a):
    """Calculate the toroidal magnetic field at the outboard midplane.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
        B0: [T] :term:`glossary link<magnetic_field_on_axis>`
        R: [m] :term:`glossary link<major_radius>`
        a: [m] :term:`glossary link<minor_radius>`

    Returns:
         B_t_out_mid [T]
    """
    # CHECK
    return B0 * (R / (R + a))


@relation(
    name="Fieldline pitch at outboard midplane",
    tags=("power_exhaust", "tokamak"),
    outputs="fieldline_pitch_at_omp",
)
def calc_fieldline_pitch_at_omp(B_t_out_mid, B_pol_out_mid):
    """Calculate the pitch of the magnetic field at the outboard midplane.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
        B_t_out_mid: [T] :term:`glossary link<B_t_out_mid>`
        B_pol_out_mid: [T] :term:`glossary link<B_pol_out_mid>`

    Returns:
         fieldline_pitch_at_omp [~]
    """
    # CHECK
    return np.sqrt(B_t_out_mid**2 + B_pol_out_mid**2) / B_pol_out_mid


@relation(
    name="Parallel heat flux density",
    tags=("power_exhaust", "tokamak"),
    outputs="q_parallel",
)
def calc_parallel_heat_flux_density(P_sep, fraction_of_P_SOL_to_divertor, R, a, lambda_q, fieldline_pitch_at_omp):
    """Calculate the parallel heat flux density entering a flux tube (q_par) at the outboard midplane.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    This expression is power to target divided by the area perpendicular to the flux tube.
    The poloidal area of a ring at the outboard midplane is 2 * pi * (R + a) * lambda_q;
    projecting the poloidal heat flux density to parallel divides by the field-line pitch.

    Args:
      P_sep: [W] :term:`glossary link<power_crossing_separatrix>`
      fraction_of_P_SOL_to_divertor: [~] :term:`glossary link<fraction_of_P_SOL_to_divertor>`
      R: [m] :term:`glossary link<major_radius>`
      a: [m] :term:`glossary link<minor_radius>`
      lambda_q: [m] :term:`glossary link<lambda_q>`
      fieldline_pitch_at_omp: [~] :term:`glossary link<fieldline_pitch_at_omp>`

    Returns:
      q_parallel [W/m^2]
    """
    # CHECK
    upstream_major_radius = R + a
    return (
        P_sep
        * fraction_of_P_SOL_to_divertor
        / (2.0 * np.pi * upstream_major_radius * lambda_q)
        * fieldline_pitch_at_omp
    )


@relation(
    name="Perpendicular heat flux density",
    tags=("power_exhaust", "tokamak"),
    outputs="q_perp",
)
def calc_q_perp(P_sep, R, a, lambda_q):
    """Calculate the perpendicular heat flux at the outboard midplane.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
      P_sep: [W] :term:`glossary link<power_crossing_separatrix>`
      R: [m] :term:`glossary link<major_radius>`
      a: [m] :term:`glossary link<minor_radius>`
      lambda_q: [m] :term:`glossary link<lambda_q>`

    Returns:
      q_perp [W/m^2]
    """
    # CHECK
    return P_sep / (2.0 * np.pi * (R + a) * lambda_q)
