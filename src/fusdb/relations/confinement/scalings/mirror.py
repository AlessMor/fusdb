"""Magnetic-mirror particle- and energy-confinement relations.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import lambertw

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J


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
