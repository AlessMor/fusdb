"""Plasma pressure relations expressed via Reactor.relation."""
from __future__ import annotations

from fusdb.registry.constants import KEV_TO_J
from fusdb.reactor_class import Reactor


@Reactor.relation("plasma", name="Thermal pressure", output="p_th")
def thermal_pressure(n_e: float, T_e: float, n_i: float, T_i: float) -> float:
    """Return thermal pressure from electron/ion densities and temperatures."""
    return (n_e * T_e + n_i * T_i) * KEV_TO_J


@Reactor.relation("plasma", name="Peak pressure", output="p_peak")
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> float:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
