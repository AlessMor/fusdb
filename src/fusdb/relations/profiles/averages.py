"""Profile average relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import line_average, volume_average


def _profile_average_residual(avg: Any, profile: Any, rho: Any, w_V: Any) -> Any:
    """Return the scale-normalized residual for ``avg == volume_average(profile)``.

    ``w_V`` is the geometry-provided volume measure on the common computational
    ``rho`` grid. Keeping the sampling coordinate and the integration measure
    separate lets the same profile relation work for different magnetic
    configurations without assigning physical meaning to bare ``rho``.
    """
    lhs = np.asarray(avg, dtype=float)
    rhs = np.asarray(volume_average(profile, rho, weight=w_V), dtype=float)
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("profile average must be finite")
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


@relation(name="Magnetic-field volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def magnetic_field_volume_average(B_avg: float, B: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(B_avg, B, rho, w_V)


@relation(name="Electron temperature volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def electron_temperature_volume_average(T_e_avg: float, T_e: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(T_e_avg, T_e, rho, w_V)


@relation(name="Ion temperature volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def ion_temperature_volume_average(T_i_avg: float, T_i: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(T_i_avg, T_i, rho, w_V)


@relation(name="Electron density volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def electron_density_volume_average(n_e_avg: float, n_e: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(n_e_avg, n_e, rho, w_V)


@relation(name="Ion density volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def ion_density_volume_average(n_fuel_avg: float, n_fuel: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(n_fuel_avg, n_fuel, rho, w_V)


@relation(name="Deuterium density volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def deuterium_density_volume_average(n_D_avg: float, n_D: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(n_D_avg, n_D, rho, w_V)


@relation(name="Tritium density volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def tritium_density_volume_average(n_T_avg: float, n_T: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(n_T_avg, n_T, rho, w_V)


@relation(name="Helium-3 density volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def helium3_density_volume_average(n_He3_avg: float, n_He3: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(n_He3_avg, n_He3, rho, w_V)


@relation(name="Helium-4 density volume-average consistency", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"))
def helium4_density_volume_average(n_He4_avg: float, n_He4: Any, w_V: Any, rho: Any) -> Any:
    return _profile_average_residual(n_He4_avg, n_He4, rho, w_V)


@relation(name="Magnetic-field rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="B_rho_avg")
def magnetic_field_rho_average(B: Any, rho: Any) -> Any:
    return line_average(B, rho)


@relation(name="Electron temperature rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="T_e_rho_avg")
def electron_temperature_rho_average(T_e: Any, rho: Any) -> Any:
    return line_average(T_e, rho)


@relation(name="Ion temperature rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="T_i_rho_avg")
def ion_temperature_rho_average(T_i: Any, rho: Any) -> Any:
    return line_average(T_i, rho)


@relation(name="Electron density rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="n_e_rho_avg")
def electron_density_rho_average(n_e: Any, rho: Any) -> Any:
    return line_average(n_e, rho)


@relation(name="Electron density line-average", tags=("plasma", "profile", "tokamak"), outputs="n_la")
def electron_density_line_average(n_e: Any, rho_minor: Any) -> Any:
    """Return the legacy tokamak radial line average using explicit ``r/a``."""
    return line_average(n_e, rho_minor)


@relation(name="Ion density rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="n_fuel_rho_avg")
def ion_density_rho_average(n_fuel: Any, rho: Any) -> Any:
    return line_average(n_fuel, rho)


@relation(name="Deuterium density rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="n_D_rho_avg")
def deuterium_density_rho_average(n_D: Any, rho: Any) -> Any:
    return line_average(n_D, rho)


@relation(name="Tritium density rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="n_T_rho_avg")
def tritium_density_rho_average(n_T: Any, rho: Any) -> Any:
    return line_average(n_T, rho)


@relation(name="Helium-3 density rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="n_He3_rho_avg")
def helium3_density_rho_average(n_He3: Any, rho: Any) -> Any:
    return line_average(n_He3, rho)


@relation(name="Helium-4 density rho-average", tags=("plasma", "profile", "tokamak", "stellarator", "mirror"), outputs="n_He4_rho_avg")
def helium4_density_rho_average(n_He4: Any, rho: Any) -> Any:
    return line_average(n_He4, rho)
