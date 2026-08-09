"""Geometry-provided mappings for the common fusdb profile grid."""

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
    """Return the legacy tokamak mapping ``r/a = rho``.

    This relation is intentionally an identity for the initial migration. It
    makes the physical minor-radius convention explicit without changing any
    existing tokamak numerics.
    """
    return np.asarray(rho, dtype=float).copy()


@relation(
    name="Tokamak normalized enclosed volume",
    tags=("geometry", "tokamak", "default"),
    outputs="v_norm",
    dependency="generated_profile",
)
def tokamak_normalized_enclosed_volume(*, rho: Any) -> np.ndarray:
    """Return the initial self-similar tokamak enclosed-volume mapping.

    ``v_norm = rho**2`` is the migration-compatible reduced geometry. Exact
    geometry-dependent enclosed-volume mappings can replace this provider later
    without changing consumers.
    """
    x = np.asarray(rho, dtype=float)
    return x**2


@relation(
    name="Tokamak volume integration weight",
    tags=("geometry", "tokamak", "default"),
    outputs="w_V",
    dependency="generated_profile",
)
def tokamak_volume_integration_weight(*, rho: Any) -> np.ndarray:
    """Return the legacy-equivalent volume weight ``w_V = rho``.

    The normalization is immaterial. This choice reproduces fusdb's historical
    discrete volume average exactly: integral(f rho d rho) / integral(rho d rho).
    """
    return np.asarray(rho, dtype=float).copy()
