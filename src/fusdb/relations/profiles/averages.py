"""Profile average relations."""

from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from fusdb import relation


def _volume_average(profile: Any, rho: Any) -> Any:
    """Return the rho-weighted grid average of a profile.

    Mirrors :meth:`RelationSystem._profile_average`: a trapezoidal average over the
    ``rho`` grid when it is usable, otherwise the arithmetic mean.  Keeping the two
    identical means these consistency residuals measure the same average the solver
    reconstructs profiles against.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    r = np.asarray(rho, dtype=float)
    if r.ndim == 1 and r.size == arr.size and r.size > 1:
        width = float(r[-1] - r[0])
        if width > 0.0:
            return trapezoid(arr, x=r) / width
    return np.mean(arr)


def _profile_average_residual(avg: Any, profile: Any, rho: Any) -> Any:
    """Return the scale-normalized residual for ``avg == volume_average(profile)``.

    Outputless (adirectional) by design, like ``Energy confinement balance``: the
    profile determines its average when the profile is supplied, while reconcile is
    free to move the average when the profile level is a solver degree of freedom.
    When the profile is reconstructed as ``avg * shape`` (shape mean == 1) the
    residual is identically zero, so it never fights a free-level profile.
    """
    lhs = np.asarray(avg, dtype=float)
    rhs = np.asarray(_volume_average(profile, rho), dtype=float)
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("profile average must be finite")
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


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
    name="Line averaged density from average density",
    tags=("plasma", "confinement", "tokamak"),
    outputs="n_la",
)
def line_averaged_density_from_average_density(n_avg: float) -> float:
    """Approximate line-averaged density from volume-averaged density.

    NOTE: This is a temporary bridge so confinement scalings that require
    ``n_la`` can be reached when only ``n_avg`` is supplied.  It should be
    replaced by a proper line-average relation from a density profile and
    geometry, or by reactor-specific profile-shape information.

    Args:
        n_avg: Average plasma density.

    Returns:
        Approximate line-averaged density.
    """
    return n_avg
