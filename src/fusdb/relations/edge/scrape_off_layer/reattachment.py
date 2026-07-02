"""Divertor reattachment-timescale relations.

cfspopcon's ``calc_neutral_pressure_kallenbach`` (target neutral pressure) is
intentionally not imported yet: it mixes 1e19 densities, mm lengths, eV heat-flux,
degrees and an eV-based ``kappa_ez`` of unstated unit, and could not be verified
against cfspopcon here -- it needs a careful unit derivation first. ``target_neutral_pressure``
is therefore consumed as an input below.
"""

from typing import Any

import numpy as np
from scipy import constants as scipy_constants

from fusdb import relation
from fusdb.registry import ATOMIC_MASS_UNIT_KG

_N20 = 1.0e20  # cfspopcon normalisation density


@relation(
    name="Ionization volume from AUG",
    tags=("power_exhaust", "tokamak"),
    outputs="ionization_volume",
)
def calc_ionization_volume_from_AUG(V_p: Any) -> Any:
    """Divertor ionization volume, scaled from AUG by plasma volume.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    AUG ionization volume per :cite:`henderson2024comparison`.
    """
    # CHECK
    return V_p / 13.0 * 0.4


@relation(
    name="Neutral flux density factor",
    tags=("power_exhaust", "tokamak"),
    outputs="neutral_flux_density_factor",
)
def calc_neutral_flux_density_factor(
    afuel: Any, ratio_of_molecular_to_ion_mass: Any = 2.0, wall_temperature: Any = 300.0
) -> Any:
    """Factor converting a neutral flux density to a neutral pressure.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    ``wall_temperature`` is in K; the neutral energy is k_B * wall_temperature.
    """
    # CHECK
    wall_energy = scipy_constants.k * wall_temperature  # [J]
    atoms_per_molecule = 2.0
    test_molecular_density = 1.0e20  # [m^-3]
    test_molecular_pressure = test_molecular_density * wall_energy  # [Pa]
    neutral_density = atoms_per_molecule * test_molecular_density
    molecular_mass = afuel * ratio_of_molecular_to_ion_mass * ATOMIC_MASS_UNIT_KG  # [kg]
    mean_thermal_velocity = np.sqrt(8.0 / np.pi * wall_energy / molecular_mass)  # [m/s]
    onesided_maxwellian_flux_density = 0.25 * mean_thermal_velocity
    return neutral_density * onesided_maxwellian_flux_density / test_molecular_pressure


@relation(
    name="Reattachment time (Henderson)",
    tags=("power_exhaust", "tokamak"),
    outputs="reattachment_time",
)
def calc_reattachment_time_henderson(
    target_neutral_pressure: Any,
    target_electron_density: Any,
    parallel_connection_length: Any,
    separatrix_power_transient: Any,
    ionization_volume_density_factor: Any,
    ionization_volume: Any,
) -> Any:
    """Detachment-front reattachment time, normalised to AUG.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    Equation 5 from :cite:`henderson2024comparison`.
    """
    # CHECK
    ionization_volume_average_density = ionization_volume_density_factor * target_electron_density
    term1 = target_neutral_pressure / 2.0          # / (2 Pa)
    term2 = ionization_volume_average_density / (3.0 * _N20)
    term3 = ionization_volume / 0.4                 # / (0.4 m^3)
    term4 = parallel_connection_length / 12.0       # / (12 m)
    term5 = 2.0e6 / separatrix_power_transient      # (2 MW) / P
    return 0.09 * term1 * term2 * term3 * term4 * term5
