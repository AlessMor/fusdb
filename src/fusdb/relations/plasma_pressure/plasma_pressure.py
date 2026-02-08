"""Plasma pressure relations expressed via @Relation decorators."""
from __future__ import annotations

from fusdb.registry import KEV_TO_J
from fusdb.relation_class import Relation_decorator as Relation
@Relation(name="Thermal pressure", output="p_th", tags=("plasma",))
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float) -> float:
    """Return thermal pressure from electron/ion densities and temperatures."""
    return (n_e * T_e + n_i * T_i) * KEV_TO_J


########################################
@Relation(name="Peak pressure", output="p_peak", tags=("plasma",))
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> float:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J

