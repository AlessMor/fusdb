"""Safety-factor geometry relations."""

import numpy as np

from fusdb.relation import relation
from fusdb.registry import MU0

# used in tokamaks - represents the number of toroidal turns a field line must complete to achieve a single poloidal transit
# usually q>1
#
# cfspopcon's calc_plasma_current_from_qstar is intentionally NOT imported: it is
# the algebraic inverse of "Edge safety factor q_star" below, which fusdb solves
# acausally (I_p <-> qstar), so a separate reformulation would be redundant.


@relation(
    name="Plasma shaping function for q_star",
    tags=("plasma", "stability", "tokamak"),
    outputs="f_shaping",
)
def calc_f_shaping_for_qstar(eps, kappa_areal, delta_95):
    """Calculate the shaping function.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation A11 from ITER Physics Basis Ch. 1. Eqn. A-11 :cite:`editors_iter_1999`
    See following discussion for how this function is used.
    q_95 = 5 * minor_radius^2 * magnetic_field_on_axis / (R * plasma_current) f_shaping

    Args:
        eps: [~] :term:`glossary link<inverse_aspect_ratio>`
        kappa_areal: [~] :term:`glossary link<areal_elongation>`
        delta_95: [~] :term:`glossary link<triangularity_psi95>`

    Returns:
        f_shaping [~]
    """
    # CHECK
    return ((1.0 + kappa_areal**2.0 * (1.0 + 2.0 * delta_95**2.0 - 1.2 * delta_95**3.0)) / 2.0) * (
        (1.17 - 0.65 * eps) / (1.0 - eps**2.0) ** 2.0
    )


@relation(
    name="Plasma shaping function for q_star (PROCESS IPDG89)",
    tags=("plasma", "stability", "tokamak", "process"),
    outputs="f_shaping",
)
def calc_f_shaping_for_qstar_process(eps, kappa_95, delta_95):
    """Shaping function on the 95% flux surface, PROCESS's IPDG89 convention.

    Identical algebra to :func:`calc_f_shaping_for_qstar` -- ITER Physics Basis
    Ch. 1 Eq. A-11 -- but fed the 95%-surface elongation rather than the AREAL
    one.  That is the distinction: PROCESS's ``i_plasma_current = 4`` evaluates
    this fit at ``kappa_95``/``triang95``, while fusdb's default (from cfspopcon)
    evaluates it at ``kappa_areal``/``delta_95``.  The two elongations genuinely differ
    -- 1.6518 vs 1.7188 at the large-tokamak design point -- and since the fit
    goes as roughly kappa^2, the resulting f_shaping differs by ~8%, which lands
    directly on ``I_p`` through the q_star relation.

    Gated (see the ``f_shaping`` whitelist in variables.yaml): fusdb's areal form
    stays the default.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return calc_f_shaping_for_qstar.func(eps=eps, kappa_areal=kappa_95, delta_95=delta_95)


@relation(
    name="Edge safety factor q_star",
    tags=("plasma", "stability", "tokamak"),
    outputs="qstar",
)
def calc_q_star_from_plasma_current(B0, R, eps, I_p, f_shaping):
    """Calculate an analytical estimate for the edge safety factor q_star.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Updated formula from ITER Physics Basis Ch. 1. :cite:`editors_iter_1999`

    Args:
        B0: [T] :term:`glossary link<magnetic_field_on_axis>`
        R: [m] :term:`glossary link<major_radius>`
        eps: [~] :term:`glossary link<inverse_aspect_ratio>`
        I_p: [A] :term:`glossary link<plasma_current>`
        f_shaping: [~] :term:`glossary link<f_shaping>`

    Returns:
        qstar [~]
    """
    # CHECK
    plasma_current = I_p / 1.0e6  # cfspopcon expresses plasma_current in MA
    return 5.0 * (eps * R) ** 2.0 * B0 / (plasma_current * R) * f_shaping


@relation(
    name="Cylindrical edge safety factor",
    tags=("plasma", "stability"),
    outputs="q_cyl",
)
def calc_cylindrical_edge_safety_factor(R, a, kappa_95, delta_95, B0, I_p):
    """Calculate the edge safety factor, following the formula used in the SepOS paper.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation K.6 from :cite:`Eich_2021`. Should use kappa_95 and delta_95 values.
    Gives a slightly different result to our standard q_star calculation.
    plasma_current (I_p) in A.
    """
    # CHECK
    shaping_correction = np.sqrt(
        (1.0 + kappa_95**2 * (1.0 + 2.0 * delta_95**2 - 1.2 * delta_95**3)) / 2.0
    )
    poloidal_circumference = 2.0 * np.pi * a * shaping_correction
    average_B_pol = MU0 * I_p / poloidal_circumference
    return B0 / average_B_pol * a / R * shaping_correction
