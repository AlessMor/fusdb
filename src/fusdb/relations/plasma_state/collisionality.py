"""Collisionality, gyroradius, and related plasma metrics."""

from typing import Any

import numpy as np
from scipy import constants as scipy_constants

from fusdb.relation import relation
from fusdb.registry import ATOMIC_MASS_UNIT_KG, ELECTRON_CHARGE_C, KEV_TO_J


def calc_coulomb_logarithm(electron_density: Any, electron_temp: Any) -> Any:
    """Calculate the Coulomb logarithm, for electron-electron or electron-ion collisions.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    From text on page 6 of :cite:`Verdoolaege_2021`
    """
    return 30.9 - np.log(electron_density**0.5 * electron_temp**-1.0)


@relation(
    name="Larmor radius",
    tags=("plasma", "collisionality"),
    outputs="rho_L",
)
def calc_larmor_radius(T_i_avg: Any, B0: Any, afuel: Any) -> Any:
    """Calculate the Larmor radius.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 1 from :cite:`Eich_2020`
    """
    return np.sqrt(T_i_avg * KEV_TO_J * afuel * ATOMIC_MASS_UNIT_KG) / (
        ELECTRON_CHARGE_C * B0
    )


@relation(
    name="Normalised collisionality",
    tags=("plasma", "collisionality"),
    outputs="nu_star",
)
def calc_normalised_collisionality(
    n_avg: Any,
    T_avg: Any,
    T_i_avg: Any,
    qstar: Any,
    R: Any,
    eps: Any,
    Z_eff: Any,
) -> Any:
    """Calculate normalized collisionality.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 1c from :cite:`Verdoolaege_2021`

    Extra factor of ureg.e**2, presumably related to electron_temp**-2 for electron_temp in eV

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        q_star: [~] :term:`glossary link<q_star>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [m] :term:`glossary link<inverse_aspect_ratio>`
        z_effective: [~] :term:`glossary link<z_effective>`

    Returns:
         nu_star [~]
    """
    coulomb_log = calc_coulomb_logarithm(n_avg, T_avg * 1.0e3)
    return (
        ELECTRON_CHARGE_C**4
        / (2.0 * np.pi * 3**1.5 * scipy_constants.epsilon_0**2)
        * coulomb_log
        * n_avg
        * qstar
        * R
        * Z_eff
        / ((T_i_avg * KEV_TO_J) ** 2 * eps**1.5)
    )


@relation(
    name="Turbulence parameter alpha_t",
    tags=("plasma", "collisionality"),
    outputs="alpha_t",
)
def calc_alpha_t(
    T_sep: Any,
    q_cyl: Any,
    R: Any,
    afuel: Any,
    Z_bar: Any,
    nu_ei: Any,
    ion_to_electron_temp_ratio: Any = 1.0,
) -> Any:
    """Calculate the turbulence parameter alpha_t.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 9 from :cite:`Eich_2020`. Compared to this equation, the factor of the
    ion_to_electron_temp_ratio is added following a discussion with T. Eich.


    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        cylindrical_safety_factor: :term:`glossary link<cylindrical_safety_factor>`
        major_radius: :term:`glossary link<major_radius>`
        average_ion_mass: :term:`glossary link<average_ion_mass>`
        mean_ion_charge_state: :term:`glossary link<mean_ion_charge_state>`
        nu_ei: electron-ion collision frequency
        ion_to_electron_temp_ratio: :term:`glossary link<ion_to_electron_temp_ratio>`

    Returns:
        :term:`alpha_t`
    """
    electron_temp_j = T_sep * KEV_TO_J
    ion_mass_kg = afuel * ATOMIC_MASS_UNIT_KG
    ion_sound_speed = np.sqrt(Z_bar * electron_temp_j / ion_mass_kg)

    return (
        1.02
        * nu_ei
        / ion_sound_speed
        * (scipy_constants.electron_mass / ion_mass_kg)
        * q_cyl**2
        * R
        * (1.0 + ion_to_electron_temp_ratio / Z_bar)
    )


@relation(
    name="Edge collisionality",
    tags=("plasma", "collisionality"),
    outputs="edge_collisionality",
)
def calc_edge_collisionality(
    T_sep: Any,
    q_cyl: Any,
    R: Any,
    afuel: Any,
    Z_bar: Any,
    nu_ei: Any,
    ion_to_electron_temp_ratio: Any = 1.0,
) -> Any:
    """Calculate the edge collisionality.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 7 from :cite:`Faitsch_2023`.

    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        cylindrical_safety_factor: :term:`glossary link<cylindrical_safety_factor>`
        major_radius: :term:`glossary link<major_radius>`
        average_ion_mass: :term:`glossary link<average_ion_mass>`
        z_effective: :term:`glossary link<z_effective>`
        mean_ion_charge_state: :term:`glossary link<mean_ion_charge_state>`
        ion_to_electron_temp_ratio: :term:`glossary link<ion_to_electron_temp_ratio>`

    Returns:
        :term:`edge_collisionality`
    """
    alpha_t = calc_alpha_t(
        T_sep=T_sep,
        q_cyl=q_cyl,
        R=R,
        afuel=afuel,
        Z_bar=Z_bar,
        nu_ei=nu_ei,
        ion_to_electron_temp_ratio=ion_to_electron_temp_ratio,
    )

    return 100.0 * alpha_t / q_cyl


@relation(
    name="Electron-electron collision frequency",
    tags=("plasma", "collisionality"),
    outputs="nu_ee",
)
def calc_electron_electron_collision_freq(n_sep: Any, T_sep: Any) -> Any:
    """Calculate the electron-electron collision frequency, using equation B1 from from :cite:`Eich_2020`.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    electron_temp = T_sep * KEV_TO_J
    coulomb_log = calc_coulomb_logarithm(electron_density=n_sep, electron_temp=T_sep * 1.0e3)
    return (
        (4.0 / 3.0)
        * np.sqrt(2.0 * np.pi)
        * n_sep
        * ELECTRON_CHARGE_C**4
        * coulomb_log
        / (
            (4.0 * np.pi * scipy_constants.epsilon_0) ** 2
            * np.sqrt(scipy_constants.electron_mass)
            * electron_temp**1.5
        )
    )


@relation(
    name="Electron-ion collision frequency",
    tags=("plasma", "collisionality"),
    outputs="nu_ei",
)
def calc_electron_ion_collision_freq(nu_ee: Any, Z_eff: Any) -> Any:
    """Calculate the electron-ion collision frequency, using equation B2 from from :cite:`Eich_2020`.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    z_effective_correction = (1.0 - 0.569) * np.exp(
        -(((Z_eff - 1.0) / 3.25) ** 0.85)
    ) + 0.569

    return nu_ee * z_effective_correction * Z_eff


@relation(
    name="Normalized gyroradius",
    tags=("plasma", "collisionality"),
    outputs="rho_star",
)
def calc_rho_star(rho_L: Any, a: Any) -> Any:
    """Calculate rho* (normalized gyroradius).

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 1a from :cite:`Verdoolaege_2021`

    Args:
        average_ion_mass: [amu] :term:`glossary link<average_ion_mass>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
         rho_star [~]
    """
    return rho_L / a


@relation(name="Mirror ion thermal speed", tags=("mirror", "collisionality"), outputs="v_th_i")
def mirror_ion_thermal_speed(T_i_peak: Any, afuel: Any) -> Any:
    """Mirror ion thermal speed from the peak ion temperature.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    mass = np.asarray(afuel) * ATOMIC_MASS_UNIT_KG
    return np.sqrt(2.0 * np.asarray(T_i_peak) * KEV_TO_J / mass)


@relation(name="Mirror ion gyroradius", tags=("mirror", "collisionality"), outputs="rho_i")
def mirror_ion_gyroradius(v_th_i: Any, afuel: Any, B_c: Any, Z_i: Any = 1.0) -> Any:
    """Mirror ion gyroradius in the diamagnetically corrected central field.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    mass = np.asarray(afuel) * ATOMIC_MASS_UNIT_KG
    return mass * np.asarray(v_th_i) / (
        np.asarray(Z_i) * ELECTRON_CHARGE_C * np.asarray(B_c)
    )


@relation(name="Mirror ion mean free path", tags=("mirror", "collisionality"), outputs="lambda_ii")
def mirror_ion_mean_free_path(v_th_i: Any, tau_ii: Any) -> Any:
    """Ion-ion collisional mean free path.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return np.asarray(v_th_i) * np.asarray(tau_ii)


@relation(name="Mirror collisionality regime ratio", tags=("mirror", "collisionality"), outputs="mirror_regime_ratio")
def mirror_collisionality_ratio(lambda_ii: Any, R_mc: Any, L_c: Any) -> Any:
    """VSC Eq. (60).

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return np.asarray(lambda_ii) / (np.asarray(R_mc) * np.asarray(L_c))
