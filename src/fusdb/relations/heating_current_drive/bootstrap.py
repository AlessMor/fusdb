"""Bootstrap and inductive current-drive relations."""

import numpy as np

from fusdb.relation import relation


@relation(
    name="Bootstrap current fraction",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="f_BS",
)
def calc_bootstrap_fraction(density_peaking, ion_density_peaking, temperature_peaking, Z_eff, qstar, eps, beta_p):
    """Calculate bootstrap current fraction.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    K. Gi et al, Bootstrap current fraction scaling :cite:`gi_bootstrap_2014`
    Equation assumes q0 = 1.

    cfspopcon's ``nu_n = (ion_density_peaking + electron_density_peaking) / 2``;
    fusdb's ``density_peaking`` is the electron density peaking and
    ``ion_density_peaking`` defaults to it (so this reduces to ``density_peaking``
    when the ion and electron profiles share a peaking).

    Args:
        density_peaking: [~] :term:`glossary link<density_peaking>`
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        Z_eff: [~] :term:`glossary link<z_effective>`
        qstar: [~] :term:`glossary link<q_star>`
        eps: [~] :term:`glossary link<inverse_aspect_ratio>`
        beta_p: [~] :term:`glossary link<beta_poloidal>`

    Returns:
        f_BS [~]
    """
    # CHECK
    nu_n = (ion_density_peaking + density_peaking) / 2
    temp_delta = np.maximum(temperature_peaking - 1.0, 0.0)
    total_delta = np.maximum(temp_delta + nu_n - 1.0, 0.0)
    temp_delta_for_denominator = np.maximum(temp_delta, 1.0e-12)

    bootstrap_fraction = 0.474 * (
        total_delta**0.974
        * temp_delta_for_denominator**-0.416
        * Z_eff**0.178
        * qstar**-0.133
        * eps**0.4
        * beta_p
    )

    return bootstrap_fraction


@relation(
    name="Inductive plasma current",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="inductive_plasma_current",
)
def calc_inductive_plasma_current(I_p, f_BS):
    """Calculate the inductively-driven plasma current.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    cfspopcon assumes the bootstrap current is the only non-inductive current.
    With external current drive present, use f_NI = f_BS + f_CD in place of f_BS.

    Args:
        I_p: [A] :term:`glossary link<plasma_current>`
        f_BS: [~] :term:`glossary link<bootstrap_fraction>`

    Returns:
        inductive_plasma_current [A]
    """
    # CHECK
    return I_p * (1.0 - f_BS)
