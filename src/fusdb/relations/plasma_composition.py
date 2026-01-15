"""Ion composition relations for densities and fractions."""

from __future__ import annotations

import math
from typing import Mapping

from fusdb.reactor_class import Reactor
from fusdb.relations.fusion_power.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
)


@Reactor.relation(
    "plasma",
    name="Electron density from volume-averaged density",
    output="n_e",
    variables=("n_avg",),
)
def electron_density_from_average(n_avg: float) -> float:
    """Return electron density from volume-averaged density."""
    return n_avg


@Reactor.relation(
    "plasma",
    name="Electron density from ion fractions",
    output="n_e",
)
def electron_density_from_fractions(
    n_i: float,
    f_D: float,
    f_T: float,
    f_He3: float,
    f_He4: float,
) -> float:
    """Return electron density from ion density and ion fractions."""
    return n_i * (f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4)
# NOTE: improve the formulation so that it takes all ion fractions automatically


@Reactor.relation(
    "plasma",
    name="Deuterium density from fraction",
    output="n_D",
)
def deuterium_density(f_D: float, n_i: float) -> float:
    """Return deuterium density from ion fraction and total ion density."""
    return f_D * n_i


@Reactor.relation(
    "plasma",
    name="Tritium density from fraction",
    output="n_T",
)
def tritium_density(f_T: float, n_i: float) -> float:
    """Return tritium density from ion fraction and total ion density."""
    return f_T * n_i


@Reactor.relation(
    "plasma",
    name="Helium-3 density from fraction",
    output="n_He3",
)
def helium3_density(f_He3: float, n_i: float) -> float:
    """Return helium-3 density from ion fraction and total ion density."""
    return f_He3 * n_i


@Reactor.relation(
    "plasma",
    name="Helium-4 density from fraction",
    output="n_He4",
)
def helium4_density(f_He4: float, n_i: float) -> float:
    """Return helium-4 density from ion fraction and total ion density."""
    return f_He4 * n_i


@Reactor.relation(
    "plasma",
    name="Deuterium fraction from density",
    output="f_D",
    constraints=("n_i > 0",),
)
def deuterium_fraction(n_D: float, n_i: float) -> float:
    """Return deuterium fraction from density and total ion density."""
    return n_D / n_i


@Reactor.relation(
    "plasma",
    name="Tritium fraction from density",
    output="f_T",
    constraints=("n_i > 0",),
)
def tritium_fraction(n_T: float, n_i: float) -> float:
    """Return tritium fraction from density and total ion density."""
    return n_T / n_i


@Reactor.relation(
    "plasma",
    name="Helium-3 fraction from density",
    output="f_He3",
    constraints=("n_i > 0",),
)
def helium3_fraction(n_He3: float, n_i: float) -> float:
    """Return helium-3 fraction from density and total ion density."""
    return n_He3 / n_i


@Reactor.relation(
    "plasma",
    name="Helium-4 fraction from density",
    output="f_He4",
    constraints=("n_i > 0",),
)
def helium4_fraction(n_He4: float, n_i: float) -> float:
    """Return helium-4 fraction from density and total ion density."""
    return n_He4 / n_i


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_D)",
    output="f_D",
)
def deuterium_fraction_normalized(f_T: float, f_He3: float, f_He4: float) -> float:
    """Return deuterium fraction from the ion fraction normalization."""
    return 1.0 - f_T - f_He3 - f_He4


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_T)",
    output="f_T",
)
def tritium_fraction_normalized(f_D: float, f_He3: float, f_He4: float) -> float:
    """Return tritium fraction from the ion fraction normalization."""
    return 1.0 - f_D - f_He3 - f_He4


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_He3)",
    output="f_He3",
)
def helium3_fraction_normalized(f_D: float, f_T: float, f_He4: float) -> float:
    """Return helium-3 fraction from the ion fraction normalization."""
    return 1.0 - f_D - f_T - f_He4


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_He4)",
    output="f_He4",
)
def helium4_fraction_normalized(f_D: float, f_T: float, f_He3: float) -> float:
    """Return helium-4 fraction from the ion fraction normalization."""
    return 1.0 - f_D - f_T - f_He3


@Reactor.relation(
    "plasma",
    name="Average fuel mass number",
    output="afuel",
)
def average_fuel_mass_number(f_D: float, f_T: float, f_He3: float, f_He4: float) -> float:
    """Return average ion mass number from ion fractions."""
    return 2.0 * f_D + 3.0 * f_T + 3.0 * f_He3 + 4.0 * f_He4


_SPECIES = ("D", "T", "He3", "He4")
_FRACTION_VARS = {"D": "f_D", "T": "f_T", "He3": "f_He3", "He4": "f_He4"}


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
