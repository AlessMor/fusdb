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
    outputs="n_i",
)
def default_ion_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


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
    outputs="n_i_avg",
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
