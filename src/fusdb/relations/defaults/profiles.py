"""Default profile relation helpers.

Last-resort fallbacks: a uniform profile at the average value, an average that
equals the plasma average, a peaking inherited from the electron one.  None of
that carries device physics, so they are deliberately NOT scoped to a `device`
tag.  Enumerating (tokamak, stellarator, mirror) made them unreachable for any
scenario whose device tag is not one of those three -- such a reactor then had no
1-D profiles at all, which pruned everything downstream of them (thermal
pressure, stored energy, bremsstrahlung, the whole fusion-power chain).
"""

import numpy as np
from numpy.typing import NDArray

from fusdb.relation import relation


@relation(
    name="Default ion temperature profile from average temperature",
    tags=("default", "plasma", "profile"),
    outputs="T_i",
)
def default_ion_temperature_profile_from_average_temperature(T_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-temperature profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(T_avg), dtype=float)


@relation(
    name="Default electron temperature profile from average temperature",
    tags=("default", "plasma", "profile"),
    outputs="T_e",
)
def default_electron_temperature_profile_from_average_temperature(T_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform electron-temperature profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(T_avg), dtype=float)


@relation(
    name="Default ion density profile from average density",
    tags=("default", "plasma", "profile"),
    outputs="n_fuel",
)
def default_ion_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


@relation(
    name="Default mean ion charge profile from average",
    tags=("default", "plasma", "profile"),
    outputs="Zbar_i",
)
def default_mean_ion_charge_profile_from_average(Zbar_i_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform mean-ion-charge profile Zbar_i(rho) = Zbar_i_avg.

    With radially-constant species fractions the mean ion charge Zbar_i is flat,
    so the profile is the composition scalar broadcast over rho.  This is what
    lets quasineutrality ``n_fuel = n_e/Zbar_i`` be a pointwise relation while the
    composition stays scalar; when species densities carry their own radial
    shapes a stronger Zbar_i producer gives it a real profile.
    """
    return np.full_like(np.asarray(rho, dtype=float), float(Zbar_i_avg), dtype=float)


@relation(
    name="Default electron density profile from average density",
    tags=("default", "plasma", "profile"),
    outputs="n_e",
)
def default_electron_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform electron-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


@relation(
    name="Default ion temperature average from plasma average temperature",
    tags=("default", "plasma", "profile"),
    outputs="T_i_avg",
)
def default_ion_temperature_average_from_plasma_average_temperature(T_avg: float) -> float:
    return T_avg


@relation(
    name="Default ion density average from plasma average density",
    tags=("default", "plasma", "profile"),
    outputs="n_fuel_avg",
)
def default_ion_density_average_from_plasma_average_density(n_avg: float) -> float:
    return n_avg


@relation(
    name="Default ion density peaking from electron density peaking",
    tags=("default", "plasma", "profile"),
    outputs="ion_density_peaking",
)
def default_ion_density_peaking_from_electron(density_peaking: float) -> float:
    """Fallback: ion density profile shares the electron density peaking."""
    return density_peaking


@relation(
    name="Default ion temperature peaking from electron temperature peaking",
    tags=("default", "plasma", "profile"),
    outputs="ion_temperature_peaking",
)
def default_ion_temperature_peaking_from_electron(temperature_peaking: float) -> float:
    """Fallback: ion temperature profile shares the electron temperature peaking."""
    return temperature_peaking


# The species peakings are equal BY DEFAULT in both directions.  The ion and
# electron profiles are decoupled in principle -- either may be set apart by a
# relation that has a reason to (a supplied peak/average pair, the Angioni
# scaling, a supplied profile) -- but absent such a reason the equality holds
# whichever side the scenario happens to declare.  Providers are registered by
# their declared output, so an adirectional equality needs one relation per
# direction; the default-activation gate keeps them mutually exclusive, since
# each activates only when its own output is not otherwise derivable.
#
# Without the reverse direction the inheritance was silently one-way: a
# scenario declaring only `ion_temperature_peaking` left the ELECTRON profile
# at the uniform fallback, so `temperature_peaking` derived from it as exactly
# 1.0 -- which is a singular point of the Gi bootstrap scaling (see TODO).


# DENSITY IS DELIBERATELY LEFT ONE-WAY.  The reverse (electron from ion) was
# MEASURED and REVERTED 2026-08-13: SPARC declares `nu_n_i`, so its electron
# density peaking began inheriting from it and moved every fusion-power quantity
# (11 tests).  The Angioni scalings give the two species independent density
# peakings on purpose, so equality is not the right default here; temperature has
# no such independent producer, which is why only temperature gets the pair.


@relation(
    name="Default electron temperature peaking from ion temperature peaking",
    tags=("default", "plasma", "profile"),
    outputs="temperature_peaking",
)
def default_electron_temperature_peaking_from_ion(ion_temperature_peaking: float) -> float:
    """Fallback: electron temperature profile shares the ion temperature peaking."""
    return ion_temperature_peaking
