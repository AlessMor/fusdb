"""Reduced stellarator geometry relations.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
ISS04 confinement and the Sudo density limit already exist elsewhere in FusDB;
this module supplies only the geometry quantities the reduced model uses around
those models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.numerics import volume_average
from fusdb.relation import relation


@relation(name="Stellarator volume-equivalent minor radius", tags=("stellarator", "geometry"), outputs="a_vol")
def stellarator_volume_equivalent_minor_radius(V_p: Any, R: Any) -> Any:
    """Equivalent circular-torus minor radius from V=2*pi^2 R a_vol^2."""
    return np.sqrt(np.asarray(V_p) / (2.0 * np.pi**2 * np.asarray(R)))


@relation(name="Near-axis stellarator B2.5 moment", tags=("stellarator", "geometry"), outputs="G_B25")
def stellarator_near_axis_b25_moment(
    eta_bar: Any,
    a_vol: Any,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Flux-surface-average the first-order near-axis field B/B0=1+eta_bar*r*cos(theta)."""
    x = np.asarray(rho, dtype=float)
    eta = np.asarray(eta_bar, dtype=float)
    a = np.asarray(a_vol, dtype=float)
    if eta.ndim > 0 and eta.shape[-1] != 1:
        eta = eta[..., None]
        a = a[..., None]
    theta = np.linspace(0.0, 2.0 * np.pi, 257)
    amplitude = eta[..., None] * a[..., None] * x[..., None]
    ratio = 1.0 + amplitude * np.cos(theta)
    shell_average = np.trapz(np.abs(ratio) ** 2.5, theta, axis=-1) / (2.0 * np.pi)
    return volume_average(shell_average, x, weight=w_V)
