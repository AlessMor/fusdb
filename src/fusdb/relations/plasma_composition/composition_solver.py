"""
Steady-state plasma composition solver.

Computes consistent ion density fractions (f_D, f_T, f_He3, f_He4)
accounting for fusion reaction rates and particle confinement times.

INPUT PARAMETERS:
    n_i     : Total ion density [m^-3]
    T_avg   : Volume-averaged temperature [keV]
    f_D     : Deuterium fraction (initial guess)
    f_T     : Tritium fraction (initial guess)
    f_He3   : Helium-3 fraction (initial or 0)
    f_He4   : Helium-4 fraction (initial or 0)
    tau_p   : Global particle confinement time [s]

OUTPUT PARAMETERS:
    f_D, f_T, f_He3, f_He4 : Steady-state fractions accounting for ash

BEHAVIOR:
    - If tau_p is not provided or <= 0: Returns input fractions unchanged
    - If tau_p > 0: Computes steady-state fractions with ash buildup

This solver is called:
1. During reactor initialization (in reactor_defaults.py)
2. During solver cycles when fractions may need recomputation
"""

from __future__ import annotations

import math

from fusdb.relations.power_balance.fusion_power.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
)


_SPECIES = ("D", "T", "He3", "He4")
_FRACTION_KEYS = ("f_D", "f_T", "f_He3", "f_He4")


def solve_steady_state_composition(
    n_i: float,
    T_avg: float,
    fractions: dict[str, float] | None = None,
    tau_p: float | None = None,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Solve for steady-state ion fractions.
    
    Args:
        n_i: Total ion density [m^-3]
        T_avg: Volume-averaged temperature [keV]
        fractions: Input fractions {f_D, f_T, f_He3, f_He4}
        tau_p: Particle confinement time [s]
        
    Returns:
        dict with f_D, f_T, f_He3, f_He4
        
    If tau_p is None or <= 0, returns input fractions unchanged.
    """
    # Parse input fractions
    f_D_in = fractions.get("f_D", 0.5) if fractions else 0.5
    f_T_in = fractions.get("f_T", 0.5) if fractions else 0.5
    f_He3_in = fractions.get("f_He3", 0.0) if fractions else 0.0
    f_He4_in = fractions.get("f_He4", 0.0) if fractions else 0.0
    
    # Quick return if no confinement time specified
    if tau_p is None or tau_p <= 0 or not math.isfinite(tau_p):
        return {"f_D": f_D_in, "f_T": f_T_in, "f_He3": f_He3_in, "f_He4": f_He4_in}
    
    # Validate inputs
    if n_i is None or n_i <= 0 or not math.isfinite(n_i):
        return {"f_D": f_D_in, "f_T": f_T_in, "f_He3": f_He3_in, "f_He4": f_He4_in}
    
    T_val = float(T_avg) if T_avg and math.isfinite(T_avg) else 0.0
    
    # Get reaction rates
    if T_val <= 0:
        sigmav_dt = sigmav_ddn = sigmav_dhe3 = 0.0
    else:
        sigmav_dt = float(sigmav_DT_BoschHale(T_val))
        _, sigmav_ddn, _ = sigmav_DD_BoschHale(T_val)
        sigmav_dhe3 = float(sigmav_DHe3_BoschHale(T_val))
    
    # Convert fractions to densities
    densities = {
        "D": f_D_in * n_i,
        "T": f_T_in * n_i,
        "He3": f_He3_in * n_i,
        "He4": f_He4_in * n_i,
    }
    
    # Fuel species weights (how to redistribute remaining density)
    fuel_total = f_D_in + f_T_in
    if fuel_total <= 0:
        fuel_total = 1.0
    weights = {"D": f_D_in / fuel_total, "T": f_T_in / fuel_total}
    
    # Iterate to steady state
    for _ in range(max_iter):
        n_D, n_T, n_He3 = densities["D"], densities["T"], densities["He3"]
        
        # Reaction rates
        r_dt = n_D * n_T * sigmav_dt
        r_ddn = 0.5 * n_D**2 * sigmav_ddn
        r_dhe3 = n_D * n_He3 * sigmav_dhe3
        
        updated = dict(densities)
        
        # Ash production: He4 from D-T and D-He3, He3 from D-D
        updated["He4"] = tau_p * (r_dt + r_dhe3)
        updated["He3"] = tau_p * r_ddn
        
        # Redistribute remaining density to fuel
        ash_density = updated["He3"] + updated["He4"]
        remainder = n_i - ash_density
        if remainder < 0:
            remainder = 0.0
        
        updated["D"] = weights["D"] * remainder
        updated["T"] = weights["T"] * remainder
        
        # Check convergence
        max_change = max(
            abs(updated[s] - densities[s]) / max(abs(densities[s]), 1.0)
            for s in _SPECIES
        )
        densities = updated
        if max_change <= tol:
            break
    
    return {f"f_{s}": densities[s] / n_i for s in _SPECIES}
