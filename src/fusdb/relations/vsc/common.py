"""VSC-style common zero-dimensional accounting relations."""

from __future__ import annotations
from typing import Any
import numpy as np
from fusdb.numerics import volume_average
from fusdb.relation import relation
from fusdb.registry import KEV_TO_J

@relation(name="Normalized volume radius (VSC)", tags=("geometry", "profile"), outputs="rho_vol")
def normalized_volume_radius_vsc(v_norm: Any) -> Any:
    return np.sqrt(np.clip(np.asarray(v_norm, dtype=float), 0.0, 1.0))

@relation(name="Thermal stored energy (VSC profile model)", tags=("plasma", "confinement"), outputs="W_th")
def thermal_stored_energy_vsc_profile(n_i_peak: Any, T_i_peak: Any, n0: Any, T0: Any, alphan: Any, alphat: Any, V_p: Any) -> Any:
    """Alternative W_th producer for VSC Eq. (13); FusDB's existing producer remains default."""
    f_nT = 1.0 / (1.0 + np.asarray(alphan) + np.asarray(alphat))
    return 1.5 * KEV_TO_J * np.asarray(V_p) * (np.asarray(n_i_peak) * np.asarray(T_i_peak) + np.asarray(n0) * np.asarray(T0)) * f_nT

@relation(name="Electron thermal stored energy", tags=("plasma", "confinement"), outputs="W_e")
def electron_thermal_stored_energy(n_e: Any, T_e: Any, V_p: Any, rho: Any, w_V: Any = None) -> Any:
    return 1.5 * KEV_TO_J * np.asarray(V_p) * volume_average(np.asarray(n_e) * np.asarray(T_e), rho, weight=w_V)

@relation(name="Ion thermal stored energy", tags=("plasma", "confinement"), outputs="W_i")
def ion_thermal_stored_energy(n_i: Any, T_i: Any, V_p: Any, rho: Any, w_V: Any = None) -> Any:
    return 1.5 * KEV_TO_J * np.asarray(V_p) * volume_average(np.asarray(n_i) * np.asarray(T_i), rho, weight=w_V)

@relation(name="Deposited charged fusion power", tags=("fusion_power", "power_balance"), outputs="P_charged_dep")
def deposited_charged_fusion_power(P_charged: Any, f_charged_dep: Any) -> Any:
    return np.asarray(P_charged) * np.asarray(f_charged_dep)

@relation(name="Required auxiliary heating (VSC)", tags=("auxiliary_power", "power_balance"), outputs="P_aux_required_raw")
def required_auxiliary_heating_vsc(P_loss: Any, P_brem: Any, P_sync: Any, P_line: Any, P_charged_dep: Any) -> Any:
    """Signed VSC Eq. (14), with P_trans represented by existing P_loss."""
    return np.asarray(P_loss) + np.asarray(P_brem) + np.asarray(P_sync) + np.asarray(P_line) - np.asarray(P_charged_dep)

@relation(name="Electron-ion equilibration power (VSC)", tags=("plasma", "power_balance"), outputs="P_ei")
def electron_ion_equilibration_power_vsc(n_i_avg: Any, T_i_avg: Any, T_e_avg: Any, tau_ei: Any, V_p: Any) -> Any:
    return 1.5 * np.asarray(n_i_avg) * (np.asarray(T_i_avg) - np.asarray(T_e_avg)) * KEV_TO_J * np.asarray(V_p) / np.asarray(tau_ei)

@relation(name="Electron-channel power balance (VSC)", tags=("plasma", "power_balance"))
def electron_channel_power_balance_vsc(P_charged_dep: Any, f_fast_ion: Any, P_ei: Any, P_aux_required_raw: Any, P_brem: Any, P_sync: Any, P_line: Any, W_e: Any, tau_E: Any, f_aux_e: Any) -> Any:
    lhs = (1.0 - np.asarray(f_fast_ion)) * np.asarray(P_charged_dep) + np.asarray(P_ei) + np.asarray(f_aux_e) * np.maximum(np.asarray(P_aux_required_raw), 0.0)
    rhs = np.asarray(P_brem) + np.asarray(P_sync) + np.asarray(P_line) + np.asarray(W_e) / np.asarray(tau_E)
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale

@relation(name="Tokamak magnetic field B2.5 moment", tags=("tokamak", "geometry", "power_balance"), outputs="G_B25")
def tokamak_magnetic_field_b25_moment(B: Any, B0: Any, rho: Any, w_V: Any = None) -> Any:
    return volume_average(np.abs(np.asarray(B, dtype=float) / np.asarray(B0, dtype=float)) ** 2.5, rho, weight=w_V)

@relation(name="Cyclotron loss from prescribed tau_C", tags=("power_balance",), outputs="P_sync")
def cyclotron_loss_from_tau_c(W_e: Any, tau_C: Any) -> Any:
    return np.asarray(W_e) / np.asarray(tau_C)
