"""Reduced magnetic-mirror geometry relations.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(name="Mirror diamagnetic central field", tags=("mirror", "geometry"), outputs="B_c")
def mirror_diamagnetic_central_field(B_vac: Any, beta: Any) -> Any:
    """VSC Eq. (53)."""
    return np.asarray(B_vac) * np.sqrt(np.maximum(1.0 - np.asarray(beta), 1.0e-15))


@relation(name="Mirror corrected mirror ratio", tags=("mirror", "geometry"), outputs="R_mc")
def mirror_corrected_ratio(R_m: Any, beta: Any) -> Any:
    """VSC Eq. (53)."""
    return np.asarray(R_m) / np.sqrt(np.maximum(1.0 - np.asarray(beta), 1.0e-15))


@relation(name="Mirror sin2 throat plasma volume", tags=("mirror", "geometry"), outputs="V_p")
def mirror_sin2_volume(a_c: Any, L_c: Any, L_th: Any, R_mc: Any) -> Any:
    """VSC Eq. (52)."""
    return np.pi * np.asarray(a_c) ** 2 * (
        np.asarray(L_c) + 2.0 * np.asarray(L_th) / np.sqrt(np.asarray(R_mc))
    )


@relation(name="Mirror throat area", tags=("mirror", "geometry"), outputs="A_th")
def mirror_throat_area(a_c: Any, beta: Any, R_m: Any) -> Any:
    """VSC Eq. (59)."""
    return np.pi * np.asarray(a_c) ** 2 * np.sqrt(np.maximum(1.0 - np.asarray(beta), 0.0)) / np.asarray(R_m)


@relation(name="Mirror axial B^2.5 moment", tags=("mirror", "geometry"), outputs="M_B25")
def mirror_axial_b25_moment(B_c: Any, a_c: Any, L_c: Any, L_th: Any, R_m: Any) -> Any:
    """Numerically evaluate VSC Eq. (63) over one throat and double it."""
    u = np.linspace(0.0, 1.0, 257)
    ratio = 1.0 + (np.asarray(R_m)[..., None] - 1.0) * np.sin(0.5 * np.pi * u) ** 2
    Bz = np.asarray(B_c)[..., None] * ratio
    area = np.pi * np.asarray(a_c)[..., None] ** 2 / ratio
    throat = np.trapz(Bz**2.5 * area, u, axis=-1) * np.asarray(L_th)
    central = np.asarray(B_c) ** 2.5 * np.pi * np.asarray(a_c) ** 2 * np.asarray(L_c)
    return central + 2.0 * throat
