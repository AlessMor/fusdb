"""Geometry-provided mappings for the common fusdb profile grid.

The relations in this module are deliberately *reduced* defaults. They make
coordinate semantics explicit without pretending to replace an equilibrium
solver: an imported equilibrium may provide the same variables directly and the
source-aware builder will then suppress these fallbacks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name="Tokamak normalized minor-radius coordinate",
    tags=("geometry", "tokamak", "default"),
    outputs="rho_minor",
    dependency="generated_profile",
)
def tokamak_normalized_minor_radius(*, rho: Any) -> np.ndarray:
    """Return the migration-compatible tokamak mapping ``r/a = rho``.

    This identity is intentional for the initial migration: it makes every
    relation that really assumes minor radius request ``rho_minor`` explicitly
    while remaining bit-compatible with the historical tokamak convention.
    A geometry/equilibrium provider may later replace it.
    """
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Tokamak normalized enclosed volume",
    tags=("geometry", "tokamak", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def tokamak_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Return the legacy-equivalent reduced tokamak mapping ``v_norm=rho**2``."""
    x = np.asarray(rho, dtype=float)
    return x**2


@relation(
    name="Tokamak volume integration weight",
    tags=("geometry", "tokamak", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def tokamak_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Return ``w_V=rho``, exactly reproducing the historical discrete average."""
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Sauter self-similar profile volume mapping",
    tags=("geometry", "tokamak", "default"),
    outputs=("v_norm", "w_V"),
    dependency="generated_profile",
)
def sauter_self_similar_profile_volume_mapping(
    delta: float,
    eps: float,
    *,
    rho: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a geometry-dependent nested-volume mapping for Sauter surfaces.

    This is an **opt-in** provider, not the tokamak default.  It assumes nested
    surfaces are self-similar Sauter cross-sections: ``a(rho)=a*rho`` while
    elongation, triangularity and squareness are held fixed.  Applying the
    existing Sauter total-volume correction

    ``V = 2*pi*R*(1 - 0.25*delta*eps)*S_phi``

    to each nested surface gives

    ``V(<rho)/V_p = rho**2 * (1 - c*rho)/(1 - c)``,
    ``c = 0.25*delta*eps``.

    ``w_V`` is one half of ``d(v_norm)/d(rho)``.  The arbitrary factor one half
    is chosen so the circular/zero-triangularity limit is exactly the historical
    ``w_V=rho``; volume averages are invariant to an overall weight factor.

    The approximation is deliberately explicit.  A real equilibrium should
    provide ``v_norm``/``w_V`` directly and supersede this provider rather than
    forcing a 3-D or flux-surface calculation into the nonlinear systems solve.
    """
    x = np.asarray(rho, dtype=float)
    c = 0.25 * np.asarray(delta, dtype=float) * np.asarray(eps, dtype=float)
    # Scalar scenario values stay scalar. Batched scalar values are promoted to
    # a trailing singleton dimension so they broadcast over the profile grid.
    if c.ndim > 0 and c.shape[-1] != 1:
        c = c[..., None]
    denom = 1.0 - c
    if not np.all(np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("Sauter nested-volume mapping requires 1 - 0.25*delta*eps > 0.")
    v_norm = x**2 * (1.0 - c * x) / denom
    w_V = (x - 1.5 * c * x**2) / denom
    if np.any(w_V < 0.0):
        raise ValueError("Sauter nested-volume mapping produced a negative volume weight.")
    return v_norm, w_V


@relation(
    name="Reduced stellarator normalized minor-radius coordinate",
    tags=("geometry", "stellarator", "default"),
    outputs="rho_minor",
    dependency="generated_profile",
)
def reduced_stellarator_normalized_minor_radius(*, rho: Any) -> np.ndarray:
    """Return a reduced stellarator normalized-minor-radius mapping.

    Some stellarator source data are published explicitly versus normalized
    minor radius (for example the GIGA profiles imported in ``reactors/GIGA``).
    Until a fixed-boundary equilibrium mapping is supplied, the reduced model
    identifies that published coordinate with the common computational grid.
    This is a compatibility fallback, not an equilibrium-derived ``r/a``.
    """
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Reduced stellarator toroidal-flux coordinate",
    tags=("geometry", "stellarator", "default"),
    outputs="rho_tor",
    dependency="generated_profile",
)
def reduced_stellarator_toroidal_flux_coordinate(*, rho: Any) -> np.ndarray:
    """Return reduced ``rho_tor=rho`` until an equilibrium mapping is supplied."""
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Reduced stellarator normalized enclosed volume",
    tags=("geometry", "stellarator", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def reduced_stellarator_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Return ``v_norm=rho**2`` as the behavior-neutral stellarator fallback."""
    x = np.asarray(rho, dtype=float)
    return x**2


@relation(
    name="Reduced stellarator volume integration weight",
    tags=("geometry", "stellarator", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def reduced_stellarator_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Return ``w_V=rho`` to preserve pre-refactor stellarator averaging exactly."""
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Reduced mirror radial coordinate",
    tags=("geometry", "mirror", "default"),
    outputs="rho_radial",
    dependency="generated_profile",
)
def reduced_mirror_radial_coordinate(*, rho: Any) -> np.ndarray:
    """Return a reduced mirror radial mapping ``rho_radial=rho``.

    This coordinate is radial only. Genuinely axial mirror physics belongs in
    separate reduced moments/scalars and must not overload the common ``rho``
    profile dimension.
    """
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Reduced mirror normalized enclosed volume",
    tags=("geometry", "mirror", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def reduced_mirror_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Return ``v_norm=rho**2`` as the initial reduced mirror volume mapping."""
    x = np.asarray(rho, dtype=float)
    return x**2


@relation(
    name="Reduced mirror volume integration weight",
    tags=("geometry", "mirror", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def reduced_mirror_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Return ``w_V=rho`` as the behavior-neutral reduced mirror fallback."""
    return np.asarray(rho, dtype=float).copy()
