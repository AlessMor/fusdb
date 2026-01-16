"""Solver for steady-state plasma composition."""

from __future__ import annotations

import math
from typing import Mapping

from fusdb.relations.fusion_power.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
)


_SPECIES = ("D", "T", "He3", "He4")
_FRACTION_VARS = {"D": "f_D", "T": "f_T", "He3": "f_He3", "He4": "f_He4"}
# TODO(low): improve it to work automatically by taking data from allowed_species


def solve_steady_state_composition(
    n_i: float,
    T_avg: float,
    fractions: Mapping[str, float] | None = None,
    tau_p: Mapping[str, float] | None = None,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Solve for steady-state ion fractions given total density, temperature, and confinement times."""
    n_i_val = float(n_i)
    if not math.isfinite(n_i_val):
        raise ValueError("n_i must be finite")
    if n_i_val < 0:
        raise ValueError("n_i must be >= 0")
    if n_i_val == 0:
        return {var: 0.0 for var in _FRACTION_VARS.values()}

    T_val = float(T_avg)
    if not math.isfinite(T_val):
        raise ValueError("T_avg must be finite")
    if T_val < 0:
        raise ValueError("T_avg must be >= 0")

    fraction_inputs: dict[str, float | None] = {species: None for species in _SPECIES}
    if fractions is not None:
        for species in _SPECIES:
            keys = (_FRACTION_VARS[species], species)
            for key in keys:
                if key in fractions:
                    fraction_inputs[species] = float(fractions[key])
                    break

    controlled: dict[str, float] = {}
    for species, value in fraction_inputs.items():
        if value is None:
            continue
        if not math.isfinite(value):
            raise ValueError(f"{_FRACTION_VARS[species]} must be finite")
        if value < 0:
            raise ValueError(f"{_FRACTION_VARS[species]} must be >= 0")
        controlled[species] = value

    if controlled:
        total = sum(controlled.values())
        if total <= 0:
            raise ValueError("Fuel fraction sum must be positive")
        weights = {species: value / total for species, value in controlled.items()}
    else:
        weights = {"D": 0.5, "T": 0.5}

    controlled_species = set(weights)
    uncontrolled_species = set(_SPECIES) - controlled_species

    tau_p_map: dict[str, float] = {}
    for species in _SPECIES:
        tau_value = None
        if tau_p is not None:
            for key in (species, f"tau_p_{species}", "tau_p", "default"):
                if key in tau_p:
                    tau_value = float(tau_p[key])
                    break
        if tau_value is None:
            tau_p_map[species] = 0.0
            continue
        if not math.isfinite(tau_value) or tau_value <= 0:
            raise ValueError(f"tau_p_{species} must be > 0")
        tau_p_map[species] = tau_value

    for species in uncontrolled_species:
        if tau_p_map[species] <= 0:
            raise ValueError(f"tau_p_{species} is required for steady-state composition")

    if T_val <= 0:
        sigmav_dt = 0.0
        sigmav_ddn = 0.0
        sigmav_ddp = 0.0
        sigmav_dhe3 = 0.0
    else:
        sigmav_dt = float(sigmav_DT_BoschHale(T_val))
        _, sigmav_ddn, sigmav_ddp = sigmav_DD_BoschHale(T_val)
        sigmav_dhe3 = float(sigmav_DHe3_BoschHale(T_val))

    densities: dict[str, float] = {species: 0.0 for species in _SPECIES}
    for species, weight in weights.items():
        densities[species] = weight * n_i_val

    for _ in range(max_iter):
        n_D = densities["D"]
        n_T = densities["T"]
        n_He3 = densities["He3"]

        r_dt = n_D * n_T * sigmav_dt
        r_ddn = 0.5 * n_D**2 * sigmav_ddn
        r_ddp = 0.5 * n_D**2 * sigmav_ddp
        r_dhe3 = n_D * n_He3 * sigmav_dhe3

        updated = dict(densities)
        if "T" in uncontrolled_species:
            denom = n_D * sigmav_dt + 1.0 / tau_p_map["T"]
            updated["T"] = r_ddp / denom if denom > 0 else 0.0
        if "He3" in uncontrolled_species:
            denom = n_D * sigmav_dhe3 + 1.0 / tau_p_map["He3"]
            updated["He3"] = r_ddn / denom if denom > 0 else 0.0
        if "He4" in uncontrolled_species:
            updated["He4"] = tau_p_map["He4"] * (r_dt + r_dhe3)

        remainder = n_i_val - sum(updated[species] for species in uncontrolled_species)
        if remainder < 0:
            raise ValueError("Total ion density is smaller than steady-state ash density")
        for species, weight in weights.items():
            updated[species] = weight * remainder

        max_change = 0.0
        for species in _SPECIES:
            prev = densities[species]
            curr = updated[species]
            scale = max(abs(prev), abs(curr), 1.0)
            max_change = max(max_change, abs(curr - prev) / scale)
        densities = updated
        if max_change <= tol:
            break

    fractions_out = {
        _FRACTION_VARS[species]: value / n_i_val for species, value in densities.items()
    }
    return fractions_out
