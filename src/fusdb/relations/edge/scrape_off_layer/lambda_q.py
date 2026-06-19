"""Scrape-off-layer lambda-q relations."""

def calc_lambda_q_with_eich_regression_15(
    power_crossing_separatrix, major_radius, B_pol_out_mid, inverse_aspect_ratio, lambda_q_factor=1.0
):
    """cfspopcon: lambda_q from Eich regression #15 (Eich 2013, Table 3)."""
    lambda_q = 1.35 * major_radius**0.04 * B_pol_out_mid**-0.92 * inverse_aspect_ratio**0.42
    if power_crossing_separatrix > 0:
        return lambda_q_factor * lambda_q * power_crossing_separatrix**-0.02
    return lambda_q_factor * lambda_q


def calc_separatrix_electron_density(nesep_over_nebar, average_electron_density):
    """cfspopcon: separatrix electron density = nesep_over_nebar * average_electron_density."""
    return nesep_over_nebar * average_electron_density
