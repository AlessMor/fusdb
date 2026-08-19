"""Reduced magnetic-mirror relations following VSC section 3.2."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import lambertw

from fusdb.relation import relation
from fusdb.registry import ATOMIC_MASS_UNIT_KG, ELECTRON_CHARGE_C, KEV_TO_J, MU0


@relation(name="Mirror peak beta (VSC)", tags=("mirror", "plasma"), outputs="beta")
def mirror_peak_beta_vsc(pres_plasma_on_axis: Any, B_vac: Any) -> Any:
    return 2.0 * MU0 * np.asarray(pres_plasma_on_axis) / np.asarray(B_vac) ** 2


@relation(name="Mirror diamagnetic central field", tags=("mirror", "geometry"), outputs="B_c")
def mirror_diamagnetic_central_field(B_vac: Any, beta: Any) -> Any:
    """VSC Eq. (53)."""
    return np.asarray(B_vac) * np.sqrt(np.maximum(1.0 - np.asarray(beta), 1.0e-15))


@relation(name="Mirror corrected mirror ratio", tags=("mirror", "geometry"), outputs="R_mc")
def mirror_corrected_ratio(R_m: Any, beta: Any) -> Any:
    """VSC Eq. (53)."""
    return np.asarray(R_m) / np.sqrt(np.maximum(1.0 - np.asarray(beta), 1.0e-15))


@relation(name="Mirror sin2 throat plasma volume", tags=("mirror", "geometry"), outputs="V_p")
def mirror_sin2_volume(a_c: Any, L_c: Any, L_th: Any, R_mc: Any) -> Any:
    """VSC Eq. (52)."""
    return np.pi * np.asarray(a_c) ** 2 * (
        np.asarray(L_c) + 2.0 * np.asarray(L_th) / np.sqrt(np.asarray(R_mc))
    )


@relation(name="Mirror ion ambipolar barrier", tags=("mirror", "confinement"), outputs="phi_i")
def mirror_ion_ambipolar_barrier(T0: Any, R_m: Any) -> Any:
    """VSC Eq. (54), with electron central temperature T0 in keV."""
    return np.asarray(T0) * np.log(np.asarray(R_m))


@relation(name="Mirror electron ambipolar barrier", tags=("mirror", "confinement"), outputs="phi_e")
def mirror_electron_ambipolar_barrier(
    phi_i: Any,
    T0: Any,
    T_i_peak: Any,
    afuel: Any,
) -> Any:
    """Solve y exp(y)=K from VSC Eq. (55) by the principal Lambert-W branch."""
    mi_over_me = np.asarray(afuel) * 1822.888486209
    ti = np.asarray(T_i_peak)
    te = np.asarray(T0)
    ratio = np.asarray(phi_i) / np.maximum(ti, 1.0e-30)
    K = np.sqrt(mi_over_me) * (ti / te) ** 1.5 * ratio * np.exp(ratio)
    return te * np.real(lambertw(K))


@relation(name="Mirror ion thermal speed", tags=("mirror", "collisionality"), outputs="v_th_i")
def mirror_ion_thermal_speed(T_i_peak: Any, afuel: Any) -> Any:
    mass = np.asarray(afuel) * ATOMIC_MASS_UNIT_KG
    return np.sqrt(2.0 * np.asarray(T_i_peak) * KEV_TO_J / mass)


@relation(name="Mirror ion gyroradius", tags=("mirror", "collisionality"), outputs="rho_i")
def mirror_ion_gyroradius(v_th_i: Any, afuel: Any, B_c: Any, Z_i: Any = 1.0) -> Any:
    mass = np.asarray(afuel) * ATOMIC_MASS_UNIT_KG
    return mass * np.asarray(v_th_i) / (
        np.asarray(Z_i) * ELECTRON_CHARGE_C * np.asarray(B_c)
    )


@relation(name="Mirror ion mean free path", tags=("mirror", "collisionality"), outputs="lambda_ii")
def mirror_ion_mean_free_path(v_th_i: Any, tau_ii: Any) -> Any:
    return np.asarray(v_th_i) * np.asarray(tau_ii)


@relation(name="Pastukhov confinement time", tags=("mirror", "confinement"), outputs="tau_Past")
def pastukhov_confinement_time(
    tau_ii: Any,
    phi_i: Any,
    T_i_peak: Any,
    R_mc: Any,
) -> Any:
    """Strong-barrier VSC Eq. (56)."""
    r = np.asarray(phi_i) / np.asarray(T_i_peak)
    s = np.sqrt(1.0 + 1.0 / np.asarray(R_mc))
    G = s * np.log((s + 1.0) / (s - 1.0))
    x = np.asarray(T_i_peak) / (2.0 * np.asarray(phi_i))
    correction = np.maximum(1.0 + x - x**2, 1.0e-12)
    return 0.5 * np.sqrt(np.pi) * np.asarray(tau_ii) * r * np.exp(r) * G / correction


@relation(name="Gas-dynamic mirror confinement time", tags=("mirror", "confinement"), outputs="tau_gd")
def gas_dynamic_mirror_confinement_time(
    R_mc: Any,
    L_c: Any,
    v_th_i: Any,
    phi_i: Any,
    T_i_peak: Any,
) -> Any:
    """VSC Eq. (57)."""
    r = np.asarray(phi_i) / np.asarray(T_i_peak)
    return np.sqrt(np.pi) * np.asarray(R_mc) * np.asarray(L_c) * np.exp(r) / np.asarray(v_th_i)


@relation(name="Radial mirror confinement time", tags=("mirror", "confinement"), outputs="tau_rho")
def radial_mirror_confinement_time(a_c: Any, rho_i: Any, tau_ii: Any) -> Any:
    """VSC Eq. (57)."""
    return (np.asarray(a_c) / np.asarray(rho_i)) ** 2 * np.asarray(tau_ii)


@relation(name="Combined mirror particle confinement", tags=("mirror", "confinement"), outputs="tau_m")
def combined_mirror_particle_confinement(tau_Past: Any, tau_gd: Any, tau_rho: Any) -> Any:
    """VSC Eq. (58)."""
    parallel = np.asarray(tau_Past) + np.asarray(tau_gd)
    return 1.0 / (1.0 / parallel + 1.0 / np.asarray(tau_rho))


@relation(name="Mirror throat area", tags=("mirror", "geometry"), outputs="A_th")
def mirror_throat_area(a_c: Any, beta: Any, R_m: Any) -> Any:
    """VSC Eq. (59)."""
    return np.pi * np.asarray(a_c) ** 2 * np.sqrt(np.maximum(1.0 - np.asarray(beta), 0.0)) / np.asarray(R_m)


@relation(name="Mirror collisionality regime ratio", tags=("mirror", "collisionality"), outputs="mirror_regime_ratio")
def mirror_collisionality_ratio(lambda_ii: Any, R_mc: Any, L_c: Any) -> Any:
    """VSC Eq. (60)."""
    return np.asarray(lambda_ii) / (np.asarray(R_mc) * np.asarray(L_c))


@relation(name="Mirror self-consistent transport loss", tags=("mirror", "confinement", "power_balance"), outputs="P_loss")
def mirror_self_consistent_transport_loss(
    n_i_peak: Any,
    n0: Any,
    phi_i: Any,
    phi_e: Any,
    T_i_peak: Any,
    T0: Any,
    alphan: Any,
    alphat: Any,
    V_p: Any,
    tau_m: Any,
) -> Any:
    """VSC Eq. (61), emitted as the existing FusDB P_loss variable."""
    potential = (np.asarray(n_i_peak) * np.asarray(phi_i) + np.asarray(n0) * np.asarray(phi_e)) / (
        1.0 + np.asarray(alphan)
    )
    thermal = (np.asarray(n_i_peak) * np.asarray(T_i_peak) + np.asarray(n0) * np.asarray(T0)) / (
        1.0 + np.asarray(alphan) + np.asarray(alphat)
    )
    return (potential + thermal) * KEV_TO_J * np.asarray(V_p) / np.asarray(tau_m)


@relation(name="Mirror loss-cone charged deposition", tags=("mirror", "fusion_power"), outputs="f_charged_dep")
def mirror_loss_cone_charged_deposition(R_mc: Any) -> Any:
    """Optimistic solid-angle estimate in VSC Eq. (62)."""
    return np.sqrt(np.maximum(1.0 - 1.0 / np.asarray(R_mc), 0.0))


@relation(name="Mirror axial B^2.5 moment", tags=("mirror", "geometry"), outputs="M_B25")
def mirror_axial_b25_moment(B_c: Any, a_c: Any, L_c: Any, L_th: Any, R_m: Any) -> Any:
    """Numerically evaluate VSC Eq. (63) over one throat and double it."""
    u = np.linspace(0.0, 1.0, 257)
    ratio = 1.0 + (np.asarray(R_m)[..., None] - 1.0) * np.sin(0.5 * np.pi * u) ** 2
    Bz = np.asarray(B_c)[..., None] * ratio
    area = np.pi * np.asarray(a_c)[..., None] ** 2 / ratio
    throat = np.trapz(Bz**2.5 * area, u, axis=-1) * np.asarray(L_th)
    central = np.asarray(B_c) ** 2.5 * np.pi * np.asarray(a_c) ** 2 * np.asarray(L_c)
    return central + 2.0 * throat


@relation(name="Mirror throat power flux", tags=("mirror", "power_exhaust"), outputs="q_throat")
def mirror_throat_power_flux(P_loss: Any, A_th: Any) -> Any:
    """Symmetric two-ended reduced end-load diagnostic."""
    return np.asarray(P_loss) / (2.0 * np.asarray(A_th))


@relation(name="Mirror collector power flux", tags=("mirror", "power_exhaust"), outputs="q_collector")
def mirror_collector_power_flux(q_throat: Any, collector_area_ratio: Any) -> Any:
    return np.asarray(q_throat) / np.asarray(collector_area_ratio)
