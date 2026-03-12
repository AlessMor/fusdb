"""Plasma pressure relations expressed via @relation decorators."""
from __future__ import annotations

from fusdb.registry import KEV_TO_J
from fusdb.relation_util import relation
from fusdb.utils import integrate_profile_over_volume, safe_float


def _is_symbolic(value: object) -> bool:
    """Return True for sympy-like symbolic values."""
    return bool(getattr(value, "free_symbols", None) is not None)


@relation(name="Thermal pressure", output="p_th", tags=("plasma",))
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float, V_p: float) -> float:
    """Return volume-averaged thermal pressure from profile/local quantities."""
    integrand = n_e * T_e + n_i * T_i
    if _is_symbolic(integrand) or _is_symbolic(V_p):
        # For scalar symbolic placeholders the profile-average pressure reduces
        # to the local expression independent of V_p.
        return KEV_TO_J * integrand

    v_scalar = safe_float(V_p)
    if v_scalar is None or v_scalar <= 0.0:
        raise ValueError("V_p must be a positive scalar for thermal pressure integration.")

    integrated = integrate_profile_over_volume(integrand, v_scalar)
    if integrated is None:
        raise ValueError("Cannot integrate thermal-pressure profile over volume.")
    return KEV_TO_J * integrated / v_scalar


########################################
@relation(name="Peak pressure", output="p_peak", tags=("plasma",))
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> float:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
