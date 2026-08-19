"""Reduced field-reversed-configuration relations following VSC section 3.3."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import beta as beta_fn

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J, MU0


def _solve_k(xs: Any) -> np.ndarray:
    """Solve tanh(K)/K = 1 - xs^2/2 (VSC Eq. 65)."""
    x = np.asarray(xs, dtype=float)
    target = 1.0 - 0.5 * x**2
    k = np.maximum(np.sqrt(np.maximum(3.0 * (1.0 - target), 1.0e-12)), 1.0e-6)
    for _ in range(30):
        th = np.tanh(k)
        sech2 = 1.0 / np.cosh(k) ** 2
        f = th / k - target
        df = (k * sech2 - th) / k**2
        step = np.where(np.abs(df) > 1.0e-14, f / df, 0.0)
        next_k = np.maximum(k - step, 1.0e-8)
        if np.all(np.abs(next_k - k) <= 1.0e-12 * np.maximum(1.0, np.abs(k))):
            k = next_k
            break
        k = next_k
    return k


def _profile_scalar(value: Any, grid: Any) -> Any:
    arr = np.asarray(value, dtype=float)
    return arr[..., None] if arr.ndim and np.asarray(grid).ndim and arr.shape[-1] != np.asarray(grid).shape[-1] else arr


@relation(name="FRC normalized radial coordinate", tags=("frc", "geometry", "default"), outputs="rho_radial", dependency="generated_profile")
def frc_normalized_radial_coordinate(*, rho: Any) -> Any:
    return np.asarray(rho, dtype=float).copy()


@relation(name="FRC separatrix wall ratio", tags=("frc", "geometry"), outputs="x_s")
def frc_separatrix_wall_ratio(r_s: Any, r_w: Any) -> Any:
    return np.asarray(r_s) / np.asarray(r_w)


@relation(name="FRC elongation", tags=("frc", "geometry"), outputs="E_frc")
def frc_elongation(L_s: Any, r_s: Any) -> Any:
    return np.asarray(L_s) / (2.0 * np.asarray(r_s))


@relation(name="FRC rigid-rotor parameter", tags=("frc", "plasma"), outputs="K_frc")
def frc_rigid_rotor_parameter(x_s: Any) -> Any:
    return _solve_k(x_s)


@relation(name="FRC field-null peak pressure", tags=("frc", "plasma"), outputs="p_peak_frc")
def frc_field_null_peak_pressure(B_e: Any) -> Any:
    """Magnetic pressure B_e^2/(2 mu0), VSC Eq. (66)."""
    return np.asarray(B_e) ** 2 / (2.0 * MU0)


@relation(name="FRC peak ion density from pressure balance", tags=("frc", "plasma"), outputs="n_i_peak")
def frc_peak_ion_density_from_pressure_balance(p_peak_frc: Any, T_i_peak: Any, T0: Any, zeta_ne_ni: Any) -> Any:
    """VSC Eq. (66): p_m=n_im (T_i + zeta T_e)."""
    energy = (np.asarray(T_i_peak) + np.asarray(zeta_ne_ni) * np.asarray(T0)) * KEV_TO_J
    return np.asarray(p_peak_frc) / energy


@relation(name="FRC peak electron density", tags=("frc", "plasma"), outputs="n0")
def frc_peak_electron_density(n_i_peak: Any, zeta_ne_ni: Any) -> Any:
    return np.asarray(zeta_ne_ni) * np.asarray(n_i_peak)


@relation(name="FRC rigid-rotor ion density profile", tags=("frc", "profile"), outputs="n_i")
def frc_rigid_rotor_ion_density(n_i_peak: Any, K_frc: Any, rho_radial: Any) -> Any:
    x = np.asarray(rho_radial, dtype=float)
    k = _profile_scalar(K_frc, x)
    n = _profile_scalar(n_i_peak, x)
    return n / np.cosh(k * (2.0 * x**2 - 1.0)) ** 2


@relation(name="FRC rigid-rotor electron density profile", tags=("frc", "profile"), outputs="n_e")
def frc_rigid_rotor_electron_density(n0: Any, K_frc: Any, rho_radial: Any) -> Any:
    x = np.asarray(rho_radial, dtype=float)
    k = _profile_scalar(K_frc, x)
    n = _profile_scalar(n0, x)
    return n / np.cosh(k * (2.0 * x**2 - 1.0)) ** 2


@relation(name="FRC rigid-rotor signed field profile", tags=("frc", "profile", "geometry"), outputs="B_signed")
def frc_rigid_rotor_signed_field(B_e: Any, K_frc: Any, rho_radial: Any) -> Any:
    x = np.asarray(rho_radial, dtype=float)
    k = _profile_scalar(K_frc, x)
    b = _profile_scalar(B_e, x)
    return b * np.tanh(k * (2.0 * x**2 - 1.0))


@relation(name="FRC magnetic-field magnitude", tags=("frc", "geometry"), outputs="B")
def frc_magnetic_field_magnitude(B_signed: Any) -> Any:
    return np.abs(np.asarray(B_signed, dtype=float))


@relation(name="FRC G1 moment", tags=("frc", "profile"), outputs="G1_frc")
def frc_g1_moment(K_frc: Any) -> Any:
    k = np.asarray(K_frc)
    return np.tanh(k) / k


@relation(name="FRC G2 moment", tags=("frc", "profile"), outputs="G2_frc")
def frc_g2_moment(K_frc: Any) -> Any:
    k = np.asarray(K_frc)
    t = np.tanh(k)
    return (t - t**3 / 3.0) / k


@relation(name="FRC mean-field moment", tags=("frc", "geometry"), outputs="G_B_frc")
def frc_mean_field_moment(K_frc: Any) -> Any:
    k = np.asarray(K_frc)
    return np.log(np.cosh(k)) / k


@relation(name="FRC B2.5 field moment", tags=("frc", "geometry"), outputs="G_B25")
def frc_b25_moment(K_frc: Any) -> Any:
    """VSC Eq. (70), evaluated independently rather than as G_B**2.5."""
    k = np.asarray(K_frc, dtype=float)
    u = np.linspace(-1.0, 1.0, 513)
    values = np.abs(np.tanh(k[..., None] * u)) ** 2.5
    return 0.5 * np.trapz(values, u, axis=-1)


@relation(name="FRC superellipse plasma volume", tags=("frc", "geometry"), outputs="V_p")
def frc_superellipse_plasma_volume(r_s: Any, L_s: Any, p_shape_frc: Any) -> Any:
    """VSC Eq. (71)."""
    p = np.asarray(p_shape_frc)
    c_p = beta_fn(1.0 / p, 1.0 + 2.0 / p) / p
    return np.pi * np.asarray(r_s) ** 2 * np.asarray(L_s) * c_p


@relation(name="FRC Ma-Xie plasma volume", tags=("frc", "geometry"), outputs="V_p")
def frc_ma_xie_plasma_volume(r_s: Any, L_s: Any, m_shape_frc: Any) -> Any:
    """Integral of VSC Eq. (72): V=pi r_s^2 L_s m/(m+1)."""
    m = np.asarray(m_shape_frc)
    return np.pi * np.asarray(r_s) ** 2 * np.asarray(L_s) * m / (m + 1.0)


@relation(name="FRC normalized enclosed volume", tags=("frc", "geometry", "default"), outputs="v_norm", dependency="generated_profile")
def frc_normalized_enclosed_volume(*, rho: Any) -> Any:
    x = np.asarray(rho, dtype=float)
    return x**2


@relation(name="FRC volume integration weight", tags=("frc", "geometry", "default"), outputs="w_V", dependency="generated_profile")
def frc_volume_integration_weight(*, rho: Any) -> Any:
    return np.asarray(rho, dtype=float).copy()


@relation(name="FRC trapped poloidal flux", tags=("frc", "plasma"), outputs="phi_p")
def frc_trapped_poloidal_flux(B_e: Any, r_s: Any, K_frc: Any) -> Any:
    """VSC Eq. (73)."""
    return np.pi * np.asarray(B_e) * np.asarray(r_s) ** 2 * np.log(np.cosh(np.asarray(K_frc))) / (2.0 * np.asarray(K_frc))


@relation(name="FRC resistive diffusion time", tags=("frc", "confinement"), outputs="tau_eta")
def frc_resistive_diffusion_time(r_s: Any, eta_plasma: Any) -> Any:
    """VSC Eq. (73): tau_eta=mu0*r_s^2/eta."""
    return MU0 * np.asarray(r_s) ** 2 / np.asarray(eta_plasma)


@relation(name="FRC confinement-to-diffusion ratio", tags=("frc", "confinement"), outputs="tau_E_over_tau_eta")
def frc_confinement_to_diffusion_ratio(tau_E: Any, tau_eta: Any) -> Any:
    return np.asarray(tau_E) / np.asarray(tau_eta)


@relation(name="FRC kinetic parameter", tags=("frc", "stability"), outputs="s_bar")
def frc_kinetic_parameter(r_s: Any, rho_ie: Any) -> Any:
    """VSC Eq. (75)."""
    return np.asarray(r_s) / np.asarray(rho_ie)


@relation(name="FRC kinetic tilt parameter", tags=("frc", "stability"), outputs="s_over_E")
def frc_kinetic_tilt_parameter(s_bar: Any, E_frc: Any) -> Any:
    return np.asarray(s_bar) / np.asarray(E_frc)


@relation(name="FRC classical diffusion coefficient", tags=("frc", "confinement"), outputs="D_classical")
def frc_classical_diffusion_coefficient(eta_plasma: Any, n_i_avg: Any, n_e_avg: Any, T_i_avg: Any, T_e_avg: Any, B_e: Any) -> Any:
    """VSC Eq. (76), with temperatures converted from keV to joules."""
    pressure = (np.asarray(n_i_avg) * np.asarray(T_i_avg) + np.asarray(n_e_avg) * np.asarray(T_e_avg)) * KEV_TO_J
    return 2.0 * np.asarray(eta_plasma) * pressure / np.asarray(B_e) ** 2


@relation(name="FRC classical confinement bracket", tags=("frc", "confinement"), outputs="tau_classical")
def frc_classical_confinement_bracket(r_s: Any, D_classical: Any) -> Any:
    return np.asarray(r_s) ** 2 / (4.0 * np.asarray(D_classical))


@relation(name="FRC Bohm diffusion coefficient", tags=("frc", "confinement"), outputs="D_Bohm")
def frc_bohm_diffusion_coefficient(T_e_avg: Any, B_e: Any) -> Any:
    """VSC Eq. (77): D_B=T_e[eV]/(16 B_e)."""
    return 1.0e3 * np.asarray(T_e_avg) / (16.0 * np.asarray(B_e))


@relation(name="FRC Bohm confinement bracket", tags=("frc", "confinement"), outputs="tau_Bohm")
def frc_bohm_confinement_bracket(r_s: Any, D_Bohm: Any) -> Any:
    return np.asarray(r_s) ** 2 / (4.0 * np.asarray(D_Bohm))


@relation(name="FRC LSX confinement time", tags=("frc", "confinement"), outputs="tau_E")
def frc_lsx_confinement_time(E_frc: Any, x_s: Any, r_s: Any, n0: Any) -> Any:
    """VSC Eq. (74), using r_s in metres and n0 in m^-3 as published by VSC."""
    return 3.2e-15 * np.asarray(E_frc) ** 0.5 * np.asarray(x_s) ** 0.8 * np.asarray(r_s) ** 2.1 * np.asarray(n0) ** 0.6
