"""Default profile relation helpers."""

import numpy as np
from numpy.typing import NDArray

from fusdb import relation


@relation(
    name="Default line averaged density from average density",
    tags=("default", "plasma", "confinement", "tokamak", "stellarator", "mirror"),
    outputs="n_la",
)
def default_line_averaged_density_from_average_density(n_avg: float) -> float:
    """Fallback tokamak approximation: line average equals volume average."""
    return n_avg


@relation(
    name="Default ion temperature profile from average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_i",
)
def default_ion_temperature_profile_from_average_temperature(T_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-temperature profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(T_avg), dtype=float)


@relation(
    name="Default electron temperature profile from average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_e",
)
def default_electron_temperature_profile_from_average_temperature(T_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform electron-temperature profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(T_avg), dtype=float)


@relation(
    name="Default ion density profile from average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_i",
)
def default_ion_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


@relation(
    name="Default electron density profile from average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_e",
)
def default_electron_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform electron-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


@relation(
    name="Default ion temperature average from plasma average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_i_avg",
)
def default_ion_temperature_average_from_plasma_average_temperature(T_avg: float) -> float:
    return T_avg


@relation(
    name="Default ion density average from plasma average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_i_avg",
)
def default_ion_density_average_from_plasma_average_density(n_avg: float) -> float:
    return n_avg
