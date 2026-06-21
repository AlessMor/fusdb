"""Impurity composition and dilution relations."""

from typing import Any

import numpy as np

from fusdb import relation


from ..utils import _species_fraction

@relation(
    name="Integrated Imp fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_Imp",
)
def integrated_impurity_fraction_from_density_profiles(n_imp: Any, n_i: Any) -> Any:
    """Return pointwise impurity fraction from density profiles."""
    return _species_fraction(n_imp, n_i, name="f_Imp")


@relation(
    name="Impurity density from ion density and impurity fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_imp",
)
def impurity_density_from_ion_density_and_fraction(n_i: Any, f_Imp: Any) -> Any:
    """Return impurity density from total ion density and impurity fraction."""
    return n_i * f_Imp

def calc_change_in_zeff(impurity_charge_state, impurity_concentration):
    """cfspopcon: change in Z_eff = Z*(Z-1)*c_imp."""
    return impurity_charge_state * (impurity_charge_state - 1.0) * impurity_concentration


def calc_change_in_dilution(impurity_charge_state, impurity_concentration):
    """cfspopcon: change in n_fuel/n_e = Z*c_imp."""
    return impurity_charge_state * impurity_concentration


def calc_zeff_and_dilution_due_to_impurities(
    average_electron_density,
    average_electron_temp,
    impurity_concentration,
    atomic_data,
):
    """cfspopcon: impact of core impurities on Z_eff and dilution.

    Returns (impurity_charge_state, change_in_zeff, change_in_dilution, z_effective,
    dilution, summed_impurity_density, average_ion_density).
    """
    from cfspopcon.formulas.impurities.impurity_charge_state import (  # noqa: F401  # TODO: atomic data
        calc_impurity_charge_state,
    )

    starting_zeff = 1.0
    starting_dilution = 1.0

    impurity_charge_state = calc_impurity_charge_state(
        average_electron_density, average_electron_temp, impurity_concentration, atomic_data
    )
    change_in_zeff = calc_change_in_zeff(impurity_charge_state, impurity_concentration)
    change_in_dilution = calc_change_in_dilution(impurity_charge_state, impurity_concentration)

    z_effective = starting_zeff + change_in_zeff.sum(dim="dim_species")
    dilution = starting_dilution - change_in_dilution.sum(dim="dim_species")
    dilution = dilution.where(dilution >= 0, 0.0)
    summed_impurity_density = impurity_concentration.sum(dim="dim_species") * average_electron_density
    average_ion_density = dilution * average_electron_density

    return (
        impurity_charge_state,
        change_in_zeff,
        change_in_dilution,
        z_effective,
        dilution,
        summed_impurity_density,
        average_ion_density,
    )
