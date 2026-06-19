"""Separatrix shape relations."""

def calc_separatrix_elongation_from_areal_elongation(areal_elongation, elongation_ratio_sep_to_areal):
    """cfspopcon: separatrix_elongation = areal_elongation * elongation_ratio_sep_to_areal."""
    return areal_elongation * elongation_ratio_sep_to_areal


def calc_separatrix_triangularity_from_triangularity95(triangularity_psi95, triangularity_ratio_sep_to_psi95):
    """cfspopcon: separatrix_triangularity = triangularity_psi95 * triangularity_ratio_sep_to_psi95."""
    return triangularity_psi95 * triangularity_ratio_sep_to_psi95


def calc_vertical_minor_radius_from_elongation_and_minor_radius(minor_radius, separatrix_elongation):
    """cfspopcon: vertical_minor_radius = minor_radius * separatrix_elongation."""
    return minor_radius * separatrix_elongation


def calc_elongation_at_psi95_from_areal_elongation(areal_elongation, elongation_ratio_areal_to_psi95):
    """cfspopcon: elongation_psi95 = areal_elongation / elongation_ratio_areal_to_psi95."""
    return areal_elongation / elongation_ratio_areal_to_psi95
