"""Plasma pressure relations expressed via @relation decorators."""
from __future__ import annotations

from fusdb.registry import KEV_TO_J
from fusdb.relation_util import relation
from fusdb.utils import integrate_profile, safe_float


@relation(name="Thermal pressure", output="p_th", tags=("plasma",))
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float, V_p: float) -> float:
    """Return volume-averaged thermal pressure from profile/local quantities."""
    integrand = n_e * T_e + n_i * T_i
    if getattr(integrand, "free_symbols", None) is not None or getattr(V_p, "free_symbols", None) is not None:
        # For scalar symbolic placeholders the profile-average pressure reduces
        # to the local expression independent of V_p.
        return KEV_TO_J * integrand

    v_scalar = safe_float(V_p)
    if v_scalar is None or v_scalar <= 0.0:
        raise ValueError("V_p must be a positive scalar for thermal pressure integration.")

    integrated = integrate_profile(integrand, v_scalar, error_label="thermal-pressure")
    return KEV_TO_J * integrated / v_scalar


########################################
@relation(name="Peak pressure", output="p_peak", tags=("plasma",))
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> float:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
