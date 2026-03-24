"""Density-based plasma composition relations."""

from __future__ import annotations

from fusdb.registry import load_allowed_species, load_allowed_variables
from fusdb.relation_util import relation
from fusdb.utils import integrate_profile

allowed_variables, _, _ = load_allowed_variables()
TRACKED_SPECIES = tuple(
    species
    for species in load_allowed_species()
    if f"n_{species}" in allowed_variables and f"f_{species}" in allowed_variables
)


@relation(
    name="Ion density from tracked species densities",
    output="n_i",
    tags=("plasma",),
)
def ion_density_from_tracked_species_densities(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
) -> float:
    """Return the total tracked ion density profile."""
    return n_D + n_T + n_He3 + n_He4


@relation(
    name="Electron density from tracked species densities",
    output="n_e",
    tags=("plasma",),
)
def electron_density_from_tracked_species_densities(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
) -> float:
    """Return the electron density profile from tracked ion densities."""
    return n_D + n_T + 2.0 * n_He3 + 2.0 * n_He4


@relation(
    name="D density from tracked ion balance",
    output="n_D",
    tags=("plasma",),
)
def deuterium_density_from_tracked_ion_balance(
    n_i: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
) -> float:
    """Return the deuterium density profile from the tracked ion balance."""
    return n_i - n_T - n_He3 - n_He4


@relation(
    name="T density from tracked ion balance",
    output="n_T",
    tags=("plasma",),
)
def tritium_density_from_tracked_ion_balance(
    n_i: float,
    n_D: float,
    n_He3: float,
    n_He4: float,
) -> float:
    """Return the tritium density profile from the tracked ion balance."""
    return n_i - n_D - n_He3 - n_He4


@relation(
    name="He3 density from tracked ion balance",
    output="n_He3",
    tags=("plasma",),
)
def helium3_density_from_tracked_ion_balance(
    n_i: float,
    n_D: float,
    n_T: float,
    n_He4: float,
) -> float:
    """Return the helium-3 density profile from the tracked ion balance."""
    return n_i - n_D - n_T - n_He4


@relation(
    name="He4 density from tracked ion balance",
    output="n_He4",
    tags=("plasma",),
)
def helium4_density_from_tracked_ion_balance(
    n_i: float,
    n_D: float,
    n_T: float,
    n_He3: float,
) -> float:
    """Return the helium-4 density profile from the tracked ion balance."""
    return n_i - n_D - n_T - n_He3


@relation(
    name="Integrated D fraction from density profiles",
    output="f_D",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_deuterium_fraction_from_density_profiles(
    n_D: float,
    n_i: float,
) -> float:
    """Return the integrated deuterium fraction from the tracked density profiles."""
    numerator_total = integrate_profile(n_D, error_label="density")
    denominator_total = integrate_profile(n_i, error_label="density")
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    return numerator_total / denominator_total


@relation(
    name="Integrated T fraction from density profiles",
    output="f_T",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_tritium_fraction_from_density_profiles(
    n_T: float,
    n_i: float,
) -> float:
    """Return the integrated tritium fraction from the tracked density profiles."""
    numerator_total = integrate_profile(n_T, error_label="density")
    denominator_total = integrate_profile(n_i, error_label="density")
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    return numerator_total / denominator_total


@relation(
    name="Integrated He3 fraction from density profiles",
    output="f_He3",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_helium3_fraction_from_density_profiles(
    n_He3: float,
    n_i: float,
) -> float:
    """Return the integrated helium-3 fraction from the tracked density profiles."""
    numerator_total = integrate_profile(n_He3, error_label="density")
    denominator_total = integrate_profile(n_i, error_label="density")
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    return numerator_total / denominator_total


@relation(
    name="Integrated He4 fraction from density profiles",
    output="f_He4",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_helium4_fraction_from_density_profiles(
    n_He4: float,
    n_i: float,
) -> float:
    """Return the integrated helium-4 fraction from the tracked density profiles."""
    numerator_total = integrate_profile(n_He4, error_label="density")
    denominator_total = integrate_profile(n_i, error_label="density")
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    return numerator_total / denominator_total


@relation(
    name="Integrated tracked fractions",
    outputs=tuple(f"f_{species}" for species in TRACKED_SPECIES),
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_tracked_fractions(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    n_i: float,
) -> dict[str, float]:
    """Return the full integrated tracked-fraction bundle from density profiles."""
    denominator = integrate_profile(n_i, error_label="density")
    if getattr(denominator, "free_symbols", None) is None and denominator <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    return {
        "f_D": integrate_profile(n_D, error_label="density") / denominator,
        "f_T": integrate_profile(n_T, error_label="density") / denominator,
        "f_He3": integrate_profile(n_He3, error_label="density") / denominator,
        "f_He4": integrate_profile(n_He4, error_label="density") / denominator,
    }


@relation(
    name="Average fuel mass number",
    output="afuel",
    tags=("plasma",),
)
def average_fuel_mass_number(
    f_D: float,
    f_T: float,
    f_He3: float,
    f_He4: float,
) -> float:
    """Return the integrated average ion mass number."""
    species_data = load_allowed_species()
    return (
        f_D * float(species_data["D"]["atomic_mass"])
        + f_T * float(species_data["T"]["atomic_mass"])
        + f_He3 * float(species_data["He3"]["atomic_mass"])
        + f_He4 * float(species_data["He4"]["atomic_mass"])
    )
