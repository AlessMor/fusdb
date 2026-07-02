"""Separatrix operational space (SepOS) — Eich 2021 / Manz 2023.

Condition functions for the L-H transition, ideal-MHD limit and L-mode density
limit (each =1 at its boundary), plus the ion/electron sustainment powers. All
dimensionless intermediates are unit-consistent in SI; temperatures enter as keV
converted to J via ``KEV_TO_J`` (and to eV for the eV-based ``kappa_0e``).

The ``read_sepos_reference`` loader is not imported.
"""

from typing import Any

import numpy as np
from scipy import constants as scipy_constants

from fusdb import relation
from fusdb.registry import ATOMIC_MASS_UNIT_KG, ELECTRON_CHARGE_C, KEV_TO_J, MU0

_KEV_TO_EV = 1.0e3
_MM_TO_M = 1.0e-3


# ── shared helpers (cfspopcon separatrix_operational_space/shared.py) ──────────
def _electron_beta(n_e: Any, T_e_keV: Any, B: Any) -> Any:
    return 2.0 * MU0 * n_e * (T_e_keV * KEV_TO_J) / B**2


def _electron_to_ion_mass_ratio(afuel: Any) -> Any:
    return scipy_constants.electron_mass / (afuel * ATOMIC_MASS_UNIT_KG)


def _curvature_drive(perpendicular_decay_length: Any, R: Any) -> Any:
    return 2.0 * perpendicular_decay_length / R


def _squared_scale_ratio(safety_factor: Any, R: Any, perpendicular_decay_length: Any) -> Any:
    return (safety_factor * R / perpendicular_decay_length) ** 2


def _ideal_MHD_wavenumber(beta_e: Any, epsilon_hat: Any, omega_B: Any, tau_i: Any, alpha_t: Any) -> Any:
    return np.sqrt(beta_e * epsilon_hat * omega_B**1.5 * (1.0 + tau_i) / alpha_t)


def _resistive_ballooning_wavenumber(critical_alpha_MHD: Any, alpha_t: Any, omega_B: Any) -> Any:
    return np.sqrt(critical_alpha_MHD / alpha_t * np.sqrt(omega_B))


def _electromagnetic_wavenumber(beta_e: Any, mu: Any) -> Any:
    return np.sqrt(beta_e / mu)


def _electron_pressure_decay_length_Eich2021H(alpha_t: Any, poloidal_sound_larmor_radius: Any, factor: float = 3.6) -> Any:
    return 1.2 * (1.0 + factor * alpha_t**1.9) * poloidal_sound_larmor_radius


def _electron_pressure_decay_length_Manz2023L(alpha_t: Any) -> Any:
    return 17.3 * alpha_t**0.298 * _MM_TO_M  # cfspopcon returns mm


def _lambda_q_Eich2020H(alpha_t: Any, poloidal_sound_larmor_radius: Any) -> Any:
    lambda_Te = 2.1 * (1.0 + 2.1 * alpha_t**1.7) * poloidal_sound_larmor_radius
    return 2.0 / 7.0 * lambda_Te


# ── shared registered quantities ──────────────────────────────────────────────
@relation(
    name="Critical alpha_MHD",
    tags=("power_exhaust", "tokamak"),
    outputs="critical_alpha_MHD",
)
def calc_critical_alpha_MHD(kappa_95: Any, delta_95: Any) -> Any:
    """Critical alpha_MHD (Eich 2021 eq. K.5).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return kappa_95**1.2 * (1.0 + 1.5 * delta_95)


@relation(
    name="Poloidal sound Larmor radius",
    tags=("power_exhaust", "tokamak"),
    outputs="poloidal_sound_larmor_radius",
)
def calc_poloidal_sound_larmor_radius(a: Any, kappa_95: Any, delta_95: Any, I_p: Any, T_sep: Any, afuel: Any) -> Any:
    """Poloidally-averaged ion sound Larmor radius (Eich 2021 eq. K.4).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    poloidal_circumference = 2.0 * np.pi * a * (1.0 + 0.55 * (kappa_95 - 1.0)) * (1.0 + 0.08 * delta_95**2)
    B_pol_avg = MU0 * I_p / poloidal_circumference
    return np.sqrt(T_sep * KEV_TO_J * afuel * ATOMIC_MASS_UNIT_KG) / (ELECTRON_CHARGE_C * B_pol_avg)


# ── condition functions ───────────────────────────────────────────────────────
@relation(
    name="SepOS L-H transition",
    tags=("power_exhaust", "tokamak"),
    outputs="SepOS_LH_transition",
)
def calc_SepOS_LH_transition(
    n_sep: Any,
    T_sep: Any,
    R: Any,
    B0: Any,
    afuel: Any,
    critical_alpha_MHD: Any,
    alpha_t: Any,
    poloidal_sound_larmor_radius: Any,
) -> Any:
    """SepOS L-H transition condition (=1 at the transition; Eich 2021 eq. 8).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    beta_e = _electron_beta(n_sep, T_sep, B0)
    mu = _electron_to_ion_mass_ratio(afuel)
    electron_pressure_decay_length = _electron_pressure_decay_length_Eich2021H(alpha_t, poloidal_sound_larmor_radius)
    k_EM = _electromagnetic_wavenumber(beta_e, mu)
    omega_B = _curvature_drive(electron_pressure_decay_length, R)

    flow_shear_stabilisation = critical_alpha_MHD * k_EM / (1.0 + (alpha_t * k_EM / critical_alpha_MHD) ** 2)
    electron_turbulence_destabilisation = 0.5 * alpha_t
    kinetic_turbulence_destabilisation = k_EM**2 * alpha_t
    ion_turbulence_destabilisation = critical_alpha_MHD / (2.0 * k_EM**2) * np.sqrt(omega_B)
    total_destabilisation = electron_turbulence_destabilisation + ion_turbulence_destabilisation + kinetic_turbulence_destabilisation
    return flow_shear_stabilisation / total_destabilisation


@relation(
    name="SepOS ideal MHD limit",
    tags=("power_exhaust", "tokamak"),
    outputs="SepOS_MHD_limit",
)
def calc_SepOS_ideal_MHD_limit(
    n_sep: Any,
    T_sep: Any,
    R: Any,
    B0: Any,
    q_cyl: Any,
    critical_alpha_MHD: Any,
    alpha_t: Any,
    poloidal_sound_larmor_radius: Any,
    ion_to_electron_temp_ratio: Any = 1.0,
) -> Any:
    """SepOS ideal-MHD limit condition (=1 at the soft limit; Eich 2021 eq. 12).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    k_RBM_factor = np.sqrt(2.0)
    electron_pressure_decay_length = _electron_pressure_decay_length_Eich2021H(alpha_t, poloidal_sound_larmor_radius)
    omega_B = _curvature_drive(electron_pressure_decay_length, R)
    beta_e = _electron_beta(n_sep, T_sep, B0)
    epsilon_hat = _squared_scale_ratio(q_cyl, R, electron_pressure_decay_length)
    k_ideal = _ideal_MHD_wavenumber(beta_e, epsilon_hat, omega_B, ion_to_electron_temp_ratio, alpha_t)
    k_RBM = _resistive_ballooning_wavenumber(critical_alpha_MHD, alpha_t, omega_B) * k_RBM_factor
    return k_ideal / k_RBM


@relation(
    name="SepOS L-mode density limit",
    tags=("power_exhaust", "tokamak"),
    outputs="SepOS_density_limit",
)
def calc_SepOS_L_mode_density_limit(
    n_sep: Any,
    T_sep: Any,
    R: Any,
    B0: Any,
    afuel: Any,
    critical_alpha_MHD: Any,
    alpha_t: Any,
) -> Any:
    """SepOS L-mode density-limit condition (=1 at the limit; Eich 2021 eq. 3).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    electron_pressure_decay_length = _electron_pressure_decay_length_Manz2023L(alpha_t)
    omega_B = _curvature_drive(electron_pressure_decay_length, R)
    beta_e = _electron_beta(n_sep, T_sep, B0)
    mu = _electron_to_ion_mass_ratio(afuel)
    k_EM = _electromagnetic_wavenumber(beta_e, mu)
    k_RBM = _resistive_ballooning_wavenumber(critical_alpha_MHD, alpha_t, omega_B)
    return k_EM / k_RBM


# ── sustainment powers (cfspopcon separatrix_operational_space/sustainment_power.py)
@relation(
    name="Sustainment power in ion channel",
    tags=("power_exhaust", "tokamak"),
    outputs="sustainment_power_in_ion_channel",
)
def calc_power_crossing_separatrix_in_ion_channel(
    A_p: Any,
    n_sep: Any,
    T_sep: Any,
    alpha_t: Any,
    poloidal_sound_larmor_radius: Any,
    ion_heat_diffusivity: Any,
    temp_scale_length_ratio: Any = 1.0,
) -> Any:
    """Power crossing the separatrix in the ion channel (Eich 2021 section 4.1).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    lambda_q = _lambda_q_Eich2020H(alpha_t, poloidal_sound_larmor_radius)
    lambda_Te = 3.5 * lambda_q
    L_Te = lambda_Te / (T_sep * KEV_TO_J)
    L_Ti = L_Te / temp_scale_length_ratio
    return A_p * n_sep * ion_heat_diffusivity / L_Ti


@relation(
    name="Sustainment power in electron channel",
    tags=("power_exhaust", "tokamak"),
    outputs="sustainment_power_in_electron_channel",
)
def calc_power_crossing_separatrix_in_electron_channel(
    T_sep: Any,
    target_electron_temp: Any,
    q_cyl: Any,
    R: Any,
    a: Any,
    B_pol_out_mid: Any,
    B_t_out_mid: Any,
    fraction_of_P_SOL_to_divertor: Any,
    Z_eff: Any,
    alpha_t: Any,
    poloidal_sound_larmor_radius: Any,
) -> Any:
    """Power crossing the separatrix in the electron channel (Eich 2021 eq. 11).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    ``kappa_0e`` is eV-based, so temperatures are handled in eV.
    """
    # CHECK
    lambda_q = _lambda_q_Eich2020H(alpha_t, poloidal_sound_larmor_radius)
    f_Zeff = 0.672 + 0.076 * np.sqrt(Z_eff) + 0.252 * Z_eff
    kappa_0e = 2600.0 / f_Zeff  # W / (eV^3.5 m)
    L_parallel = np.pi * q_cyl * R
    A_SOL = 2.0 * np.pi * (R + a) * lambda_q * B_pol_out_mid / B_t_out_mid
    T_sep_eV = T_sep * _KEV_TO_EV
    target_eV = target_electron_temp * _KEV_TO_EV
    return (
        2.0 / 7.0 * kappa_0e * A_SOL / (L_parallel * fraction_of_P_SOL_to_divertor) * (T_sep_eV**3.5 - target_eV**3.5)
    )
