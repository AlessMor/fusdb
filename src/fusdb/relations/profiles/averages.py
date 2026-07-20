"""Profile average relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import line_average, volume_average


def _profile_average_residual(avg: Any, profile: Any, rho: Any) -> Any:
    """Return the scale-normalized residual for ``avg == volume_average(profile)``.

    Outputless (adirectional) by design, like ``Energy confinement balance``: the
    profile determines its average when the profile is supplied, while reconcile is
    free to move the average when the profile level is a solver degree of freedom.
    When the profile is reconstructed as ``avg * shape`` (shape mean == 1) the
    residual is identically zero, so it never fights a free-level profile.
    """
    lhs = np.asarray(avg, dtype=float)
    rhs = np.asarray(volume_average(profile, rho), dtype=float)
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("profile average must be finite")
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


@relation(
    name="Magnetic-field volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def magnetic_field_volume_average(B_avg: float, B: Any, rho: Any) -> Any:
    """Link the magnetic-field profile to its volume-average ``B_avg``."""
    return _profile_average_residual(B_avg, B, rho)


@relation(
    name="Electron temperature volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def electron_temperature_volume_average(T_e_avg: float, T_e: Any, rho: Any) -> Any:
    """Link the electron temperature profile to its volume-average ``T_e_avg``."""
    return _profile_average_residual(T_e_avg, T_e, rho)


@relation(
    name="Ion temperature volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def ion_temperature_volume_average(T_i_avg: float, T_i: Any, rho: Any) -> Any:
    """Link the ion temperature profile to its volume-average ``T_i_avg``."""
    return _profile_average_residual(T_i_avg, T_i, rho)


@relation(
    name="Electron density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def electron_density_volume_average(n_e_avg: float, n_e: Any, rho: Any) -> Any:
    """Link the electron density profile to its volume-average ``n_e_avg``."""
    return _profile_average_residual(n_e_avg, n_e, rho)


@relation(
    name="Ion density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def ion_density_volume_average(n_i_avg: float, n_i: Any, rho: Any) -> Any:
    """Link the ion density profile to its volume-average ``n_i_avg``."""
    return _profile_average_residual(n_i_avg, n_i, rho)


@relation(
    name="Deuterium density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def deuterium_density_volume_average(n_D_avg: float, n_D: Any, rho: Any) -> Any:
    """Link the deuterium density profile to its volume-average ``n_D_avg``."""
    return _profile_average_residual(n_D_avg, n_D, rho)


@relation(
    name="Tritium density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def tritium_density_volume_average(n_T_avg: float, n_T: Any, rho: Any) -> Any:
    """Link the tritium density profile to its volume-average ``n_T_avg``."""
    return _profile_average_residual(n_T_avg, n_T, rho)


@relation(
    name="Helium-3 density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def helium3_density_volume_average(n_He3_avg: float, n_He3: Any, rho: Any) -> Any:
    """Link the helium-3 density profile to its volume-average ``n_He3_avg``."""
    return _profile_average_residual(n_He3_avg, n_He3, rho)


@relation(
    name="Helium-4 density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def helium4_density_volume_average(n_He4_avg: float, n_He4: Any, rho: Any) -> Any:
    """Link the helium-4 density profile to its volume-average ``n_He4_avg``."""
    return _profile_average_residual(n_He4_avg, n_He4, rho)


@relation(
    name="Impurity density volume-average consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def impurity_density_volume_average(n_imp_avg: float, n_imp: Any, rho: Any) -> Any:
    """Link the generic impurity density profile to its volume-average ``n_imp_avg``."""
    return _profile_average_residual(n_imp_avg, n_imp, rho)


@relation(
    name="Magnetic-field rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="B_rho_avg",
)
def magnetic_field_rho_average(B: Any, rho: Any) -> Any:
    return line_average(B, rho)


@relation(
    name="Electron temperature rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_e_rho_avg",
)
def electron_temperature_rho_average(T_e: Any, rho: Any) -> Any:
    return line_average(T_e, rho)


@relation(
    name="Ion temperature rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_i_rho_avg",
)
def ion_temperature_rho_average(T_i: Any, rho: Any) -> Any:
    return line_average(T_i, rho)


@relation(
    name="Electron density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_e_rho_avg",
)
def electron_density_rho_average(n_e: Any, rho: Any) -> Any:
    return line_average(n_e, rho)


@relation(
    name="Electron density line-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_la",
)
def electron_density_line_average(n_e: Any, rho: Any) -> Any:
    """Return ``(1/a) integral_0^a n_e(r) dr`` from a ``rho=r/a`` profile."""
    return line_average(n_e, rho)


@relation(
    name="Ion density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_i_rho_avg",
)
def ion_density_rho_average(n_i: Any, rho: Any) -> Any:
    return line_average(n_i, rho)


@relation(
    name="Deuterium density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_D_rho_avg",
)
def deuterium_density_rho_average(n_D: Any, rho: Any) -> Any:
    return line_average(n_D, rho)


@relation(
    name="Tritium density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_T_rho_avg",
)
def tritium_density_rho_average(n_T: Any, rho: Any) -> Any:
    return line_average(n_T, rho)


@relation(
    name="Helium-3 density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_He3_rho_avg",
)
def helium3_density_rho_average(n_He3: Any, rho: Any) -> Any:
    return line_average(n_He3, rho)


@relation(
    name="Helium-4 density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_He4_rho_avg",
)
def helium4_density_rho_average(n_He4: Any, rho: Any) -> Any:
    return line_average(n_He4, rho)


@relation(
    name="Impurity density rho-average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_imp_rho_avg",
)
def impurity_density_rho_average(n_imp: Any, rho: Any) -> Any:
    return line_average(n_imp, rho)
