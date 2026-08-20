"""Field-reversed-configuration confinement-time estimates and brackets.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J, MU0


@relation(name="FRC resistive diffusion time", tags=("frc", "confinement"), outputs="tau_eta")
def frc_resistive_diffusion_time(r_s: Any, eta_plasma: Any) -> Any:
    """VSC Eq. (73): tau_eta=mu0*r_s^2/eta."""
    return MU0 * np.asarray(r_s) ** 2 / np.asarray(eta_plasma)


@relation(name="FRC confinement-to-diffusion ratio", tags=("frc", "confinement"), outputs="tau_E_over_tau_eta")
def frc_confinement_to_diffusion_ratio(tau_E: Any, tau_eta: Any) -> Any:
    return np.asarray(tau_E) / np.asarray(tau_eta)


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
