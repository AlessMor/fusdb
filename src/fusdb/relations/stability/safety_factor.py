"""Safety-factor geometry relations."""

import numpy as np

from fusdb import relation
from fusdb.registry import MU0

# From operational_limits/safety_factor.py
# used in tokamaks - represents the number of toroidal turns a field line must complete to achieve a single poloidal transit
#usually q>1

# TODO(med): from cfspopcon/plasma_current add
    # edge safety factor (q*) and cylindrical edge safety factor
    # q_95 and f_shaping


def calc_f_shaping_for_qstar(inverse_aspect_ratio, areal_elongation, triangularity_psi95):
    """cfspopcon: shaping function for q_star (ITER Physics Basis Ch.1 Eq. A-11)."""
    return ((1.0 + areal_elongation**2.0 * (1.0 + 2.0 * triangularity_psi95**2.0 - 1.2 * triangularity_psi95**3.0)) / 2.0) * (
        (1.17 - 0.65 * inverse_aspect_ratio) / (1.0 - inverse_aspect_ratio**2.0) ** 2.0
    )


def calc_q_star_from_plasma_current(magnetic_field_on_axis, major_radius, inverse_aspect_ratio, plasma_current, f_shaping):
    """cfspopcon: analytical edge safety factor q_star (ITER Physics Basis Ch.1)."""
    return (
        5.0 * (inverse_aspect_ratio * major_radius) ** 2.0 * magnetic_field_on_axis / (plasma_current * major_radius) * f_shaping
    )


@relation(
    name="Cylindrical edge safety factor",
    tags=("plasma", "stability"),
    outputs="q_cyl",
)
def calc_cylindrical_edge_safety_factor(R, a, kappa_95, delta_95, B0, I_p):
    """cfspopcon: edge safety factor following SepOS (Eich 2021 Eq. K.6). plasma_current in A."""
    shaping_correction = np.sqrt(
        (1.0 + kappa_95**2 * (1.0 + 2.0 * delta_95**2 - 1.2 * delta_95**3)) / 2.0
    )
    poloidal_circumference = 2.0 * np.pi * a * shaping_correction
    average_B_pol = MU0 * I_p / poloidal_circumference
    return B0 / average_B_pol * a / R * shaping_correction
