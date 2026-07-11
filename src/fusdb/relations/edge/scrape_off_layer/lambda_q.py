"""Scrape-off-layer lambda-q relations.

Several alternative scalings for the SOL heat-flux decay length lambda_q, all
outputting ``lambda_q``; select one per reactor via tags/include (as with the
energy-confinement scalings). cfspopcon's ``calc_lambda_q`` enum dispatcher is
intentionally not imported -- fusdb selects a scaling at the relation level.

cfspopcon returns lambda_q in mm; fusdb's ``lambda_q`` is in metres, so each
scaling converts its mm result to metres (``* 1e-3``).
"""

from fusdb.relation import relation

_PA_PER_ATM = 101325.0  # cfspopcon expresses average_total_pressure in atm
_MM_TO_M = 1.0e-3       # cfspopcon returns lambda_q in mm


@relation(
    name="SOL lambda_q Brunner",
    tags=("power_exhaust", "tokamak"),
    outputs="lambda_q",
)
def calc_lambda_q_with_brunner(p_th, lambda_q_factor=1.0):
    """Return lambda_q according to the Brunner scaling.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 4 in :cite:`brunner_2018_heat_flux`. ``p_th`` is in Pa (cfspopcon atm).
    """
    # CHECK
    average_total_pressure = p_th / _PA_PER_ATM
    return _MM_TO_M * lambda_q_factor * 0.91 * average_total_pressure**-0.48


@relation(
    name="SOL lambda_q Eich regression 9",
    tags=("power_exhaust", "tokamak"),
    outputs="lambda_q",
)
def calc_lambda_q_with_eich_regression_9(B0, qstar, P_sep, lambda_q_factor=1.0):
    """Return lambda_q according to Eich regression 9.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    #9 in Table 2 in :cite:`eich_scaling_2013`. ``P_sep`` is in W (cfspopcon MW).
    """
    # CHECK
    power_crossing_separatrix = P_sep / 1.0e6
    return _MM_TO_M * lambda_q_factor * 0.7 * B0**-0.77 * qstar**1.05 * power_crossing_separatrix**0.09


@relation(
    name="SOL lambda_q Eich regression 14",
    tags=("power_exhaust", "tokamak"),
    outputs="lambda_q",
)
def calc_lambda_q_with_eich_regression_14(B_pol_out_mid, lambda_q_factor=1.0):
    """Return lambda_q according to Eich regression 14.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    #14 in Table 3 in :cite:`eich_scaling_2013`.
    """
    # CHECK
    return _MM_TO_M * lambda_q_factor * 0.63 * B_pol_out_mid**-1.19


@relation(
    name="SOL lambda_q Eich regression 15",
    tags=("power_exhaust", "tokamak"),
    outputs="lambda_q",
)
def calc_lambda_q_with_eich_regression_15(P_sep, R, B_pol_out_mid, eps, lambda_q_factor=1.0):
    """Return lambda_q according to Eich regression 15.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    #15 in Table 3 in :cite:`eich_scaling_2013`. ``P_sep`` is in W (cfspopcon MW).
    """
    # CHECK
    power_crossing_separatrix = P_sep / 1.0e6
    lambda_q = 1.35 * R**0.04 * B_pol_out_mid**-0.92 * eps**0.42
    if power_crossing_separatrix > 0:
        return _MM_TO_M * lambda_q_factor * lambda_q * power_crossing_separatrix**-0.02
    return _MM_TO_M * lambda_q_factor * lambda_q
