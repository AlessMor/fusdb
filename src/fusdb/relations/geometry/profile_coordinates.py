"""Geometry-provided mappings for the common fusdb profile grid.

These are deliberately reduced defaults. They establish the coordinate/volume
contract for each magnetic configuration without pretending to be an external
equilibrium solver. Higher-fidelity VMEC, field-line or mirror-equilibrium
adapters can replace any provider with scenario-local ``default_relation``
selection while consumers keep the same variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation


def _identity(rho: Any) -> np.ndarray:
    return np.asarray(rho, dtype=float).copy()


def _self_similar_enclosed_volume(rho: Any) -> np.ndarray:
    x = np.asarray(rho, dtype=float)
    return x**2


def _self_similar_volume_weight(rho: Any) -> np.ndarray:
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Tokamak normalized minor-radius coordinate",
    tags=("geometry", "tokamak", "default"),
    outputs="rho_minor",
    dependency="generated_profile",
)
def tokamak_normalized_minor_radius(*, rho: Any) -> np.ndarray:
    """Return the legacy tokamak mapping ``r/a = rho``."""
    return _identity(rho)


@relation(
    name="Tokamak normalized enclosed volume",
    tags=("geometry", "tokamak", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def tokamak_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Return the migration-compatible self-similar tokamak ``v_norm=rho**2``."""
    return _self_similar_enclosed_volume(rho)


@relation(
    name="Tokamak volume integration weight",
    tags=("geometry", "tokamak", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def tokamak_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Return the legacy-equivalent volume weight ``w_V=rho``."""
    return _self_similar_volume_weight(rho)


@relation(
    name="Reduced stellarator toroidal-flux coordinate",
    tags=("geometry", "stellarator", "default"),
    outputs="rho_tor",
    dependency="generated_profile",
)
def reduced_stellarator_toroidal_flux_coordinate(*, rho: Any) -> np.ndarray:
    """Use the common grid as reduced stellarator toroidal-flux radius.

    This is the zero-dimensional fallback when no equilibrium-derived mapping
    is supplied. A VMEC/equilibrium adapter should provide ``rho_tor(rho)``
    explicitly and override this relation for nontrivial flux geometry.
    """
    return _identity(rho)


@relation(
    name="Reduced stellarator normalized enclosed volume",
    tags=("geometry", "stellarator", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def reduced_stellarator_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Fallback self-similar enclosed-volume map for reduced stellarators."""
    return _self_similar_enclosed_volume(rho)


@relation(
    name="Reduced stellarator volume integration weight",
    tags=("geometry", "stellarator", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def reduced_stellarator_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Fallback volume measure for reduced stellarators.

    ``w_V=rho`` is consistent with the fallback ``v_norm=rho**2``; its overall
    scale is immaterial because volume averages normalize by its integral.
    """
    return _self_similar_volume_weight(rho)


@relation(
    name="Reduced mirror radial coordinate",
    tags=("geometry", "mirror", "default"),
    outputs="rho_radial",
    dependency="generated_profile",
)
def reduced_mirror_radial_coordinate(*, rho: Any) -> np.ndarray:
    """Use the common grid as the reduced mirror radial coordinate.

    Axial mirror physics is intentionally not encoded in this coordinate. It
    remains represented by mirror-specific scalar/moment relations; a future
    two-dimensional equilibrium layer must not overload this radial mapping.
    """
    return _identity(rho)


@relation(
    name="Reduced mirror normalized enclosed volume",
    tags=("geometry", "mirror", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def reduced_mirror_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Fallback cylindrical/self-similar radial enclosed-volume map."""
    return _self_similar_enclosed_volume(rho)


@relation(
    name="Reduced mirror volume integration weight",
    tags=("geometry", "mirror", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def reduced_mirror_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Fallback radial volume measure for the reduced mirror model."""
    return _self_similar_volume_weight(rho)
