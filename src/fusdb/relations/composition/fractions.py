"""Species fraction relations."""

from typing import Any

from fusdb import relation
from fusdb.registry import (
    SPECIES,
)

from ..utils import _positive_denominator, _species_fraction

@relation(
    name="Integrated D fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_D",
)
def integrated_deuterium_fraction_from_density_profiles(n_D: Any, n_i: Any) -> Any:
    """Return pointwise deuterium fraction from density profiles."""
    return _species_fraction(n_D, n_i, name="f_D")


@relation(
    name="Integrated T fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_T",
)
def integrated_tritium_fraction_from_density_profiles(n_T: Any, n_i: Any) -> Any:
    """Return pointwise tritium fraction from density profiles."""
    return _species_fraction(n_T, n_i, name="f_T")


@relation(
    name="Integrated He3 fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_He3",
)
def integrated_helium3_fraction_from_density_profiles(n_He3: Any, n_i: Any) -> Any:
    """Return pointwise helium-3 fraction from density profiles."""
    return _species_fraction(n_He3, n_i, name="f_He3")


@relation(
    name="Integrated He4 fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_He4",
)
def integrated_helium4_fraction_from_density_profiles(n_He4: Any, n_i: Any) -> Any:
    """Return pointwise helium-4 fraction from density profiles."""
    return _species_fraction(n_He4, n_i, name="f_He4")


@relation(
    name="Average fuel mass number",
    tags=("plasma", "composition"),
    outputs="afuel",
)
def average_fuel_mass_number(f_D: Any, f_T: Any, f_He3: Any = 0.0) -> Any:
    """Return average fuel mass number from fuel fractions.

    ``f_He3`` defaults to zero so DT cases can compute ``afuel`` from only
    ``f_D`` and ``f_T``.  If ``f_He3`` exists in the solve namespace, it is used.
    """
    fuel_total = _positive_denominator(f_D + f_T + f_He3, name="fuel ion inventory")
    numerator = (
        f_D * float(SPECIES["D"].atomic_mass)
        + f_T * float(SPECIES["T"].atomic_mass)
        + f_He3 * float(SPECIES["He3"].atomic_mass)
    )
    return numerator / fuel_total
