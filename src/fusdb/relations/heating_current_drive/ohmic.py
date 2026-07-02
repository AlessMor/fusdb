"""Ohmic heating and resistivity relations.

cfspopcon displays P_ohmic in MW and uses MA for currents; fusdb stores SI
(W, A, V, ohm*m). The formulas are unit-consistent in SI, so no rescaling is
needed.
"""

from typing import Any

from fusdb import relation


@relation(
    name="Ohmic heating power",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="P_ohmic",
)
def calc_ohmic_power(inductive_plasma_current: Any, loop_voltage: Any) -> Any:
    """Calculate the Ohmic heating power.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
        inductive_plasma_current: [A] :term:`glossary link<inductive_plasma_current>`
        loop_voltage: [V] :term:`glossary link<loop_voltage>`

    Returns:
        P_ohmic [W]
    """
    # CHECK
    return inductive_plasma_current * loop_voltage


@relation(
    name="Spitzer loop resistivity",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="spitzer_resistivity",
)
def calc_Spitzer_loop_resistivity(T_e_avg: Any) -> Any:
    """Calculate the parallel Spitzer loop resistivity assuming the Coulomb logarithm = 17 and Z=1.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Resistivity from Wesson 2.16.2 :cite:`wesson_tokamaks_2011`.

    Args:
        T_e_avg: [keV] :term:`glossary link<average_electron_temp>`

    Returns:
        spitzer_resistivity [Ohm-m]
    """
    # CHECK
    return (2.8e-8) * (T_e_avg ** (-1.5))


@relation(
    name="Resistivity trapped-particle enhancement",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="trapped_particle_fraction",
)
def calc_resistivity_trapped_enhancement(eps: Any, resistivity_trapped_enhancement_method: int = 3) -> Any:
    """Calculate the enhancement of the plasma resistivity due to trapped particles.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Definition 1 is the denominator of eta_n (neoclassical resistivity) on p801 of Wesson :cite:`wesson_tokamaks_2011`.

    Args:
        eps: [~] :term:`glossary link<inverse_aspect_ratio>`
        resistivity_trapped_enhancement_method: [~] :term:`glossary link<resistivity_trapped_enhancement_method>`

    Returns:
        trapped_particle_fraction [~]

    Raises:
        NotImplementedError: if resistivity_trapped_enhancement_method doesn't match an implementation
    """
    # CHECK
    if resistivity_trapped_enhancement_method == 1:
        trapped_particle_fraction = 1 / ((1.0 - (eps**0.5)) ** 2.0)
    elif resistivity_trapped_enhancement_method == 2:
        trapped_particle_fraction = 2 / (1.0 - 1.31 * (eps**0.5) + 0.46 * eps)
    elif resistivity_trapped_enhancement_method == 3:
        trapped_particle_fraction = 0.609 / (0.609 - 0.785 * (eps**0.5) + 0.269 * eps)
    else:
        raise NotImplementedError(
            f"No implementation {resistivity_trapped_enhancement_method} for calc_resistivity_trapped_enhancement."
        )
    return trapped_particle_fraction


@relation(
    name="Neoclassical loop resistivity",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="neoclassical_loop_resistivity",
)
def calc_neoclassical_loop_resistivity(spitzer_resistivity: Any, Z_eff: Any, trapped_particle_fraction: Any) -> Any:
    """Calculate the neoclassical loop resistivity including impurity ions.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Wesson Section 14.10. Impact of ion charge. Impact of dilution ~ 0.9.

    Args:
        spitzer_resistivity: [Ohm-m] :term:`glossary link<spitzer_resistivity>`
        Z_eff: [~] :term:`glossary link<z_effective>`
        trapped_particle_fraction: [~] :term:`glossary link<trapped_particle_fraction>`

    Returns:
        neoclassical_loop_resistivity [Ohm-m]
    """
    # CHECK
    return spitzer_resistivity * Z_eff * 0.9 * trapped_particle_fraction


@relation(
    name="Current relaxation time",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="current_relaxation_time",
)
def calc_current_relaxation_time(R: Any, eps: Any, kappa: Any, T_e_avg: Any, Z_eff: Any) -> Any:
    """Calculate the current relaxation time.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    from :cite:`Bonoli`.

    Args:
        R: [m] :term:`glossary link<major_radius>`
        eps: [~] :term:`glossary link<inverse_aspect_ratio>`
        kappa: [~] :term:`glossary link<areal_elongation>`
        T_e_avg: [keV] :term:`glossary link<average_electron_temp>`
        Z_eff: [~] :term:`glossary link<z_effective>`

    Returns:
        current_relaxation_time [s]
    """
    # CHECK
    return 1.4 * ((R * eps) ** 2.0) * kappa * (T_e_avg**1.5) / Z_eff
