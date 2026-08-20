"""Reduced field-reversed-configuration geometry relations.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import beta as beta_fn

from fusdb.relation import relation


@relation(name="FRC normalized radial coordinate", tags=("frc", "geometry", "default"), outputs="rho_radial", dependency="generated_profile")
def frc_normalized_radial_coordinate(*, rho: Any) -> Any:
    return np.asarray(rho, dtype=float).copy()


@relation(name="FRC separatrix wall ratio", tags=("frc", "geometry"), outputs="x_s")
def frc_separatrix_wall_ratio(r_s: Any, r_w: Any) -> Any:
    return np.asarray(r_s) / np.asarray(r_w)


@relation(name="FRC elongation", tags=("frc", "geometry"), outputs="E_frc")
def frc_elongation(L_s: Any, r_s: Any) -> Any:
    return np.asarray(L_s) / (2.0 * np.asarray(r_s))


@relation(name="FRC magnetic-field magnitude", tags=("frc", "geometry"), outputs="B")
def frc_magnetic_field_magnitude(B_signed: Any) -> Any:
    return np.abs(np.asarray(B_signed, dtype=float))


@relation(name="FRC mean-field moment", tags=("frc", "geometry"), outputs="G_B_frc")
def frc_mean_field_moment(K_frc: Any) -> Any:
    k = np.asarray(K_frc)
    return np.log(np.cosh(k)) / k


@relation(name="FRC B2.5 field moment", tags=("frc", "geometry"), outputs="G_B25")
def frc_b25_moment(K_frc: Any) -> Any:
    """VSC Eq. (70), evaluated independently rather than as G_B**2.5."""
    k = np.asarray(K_frc, dtype=float)
    u = np.linspace(-1.0, 1.0, 513)
    values = np.abs(np.tanh(k[..., None] * u)) ** 2.5
    return 0.5 * np.trapz(values, u, axis=-1)


@relation(name="FRC superellipse plasma volume", tags=("frc", "geometry"), outputs="V_p")
def frc_superellipse_plasma_volume(r_s: Any, L_s: Any, p_shape_frc: Any) -> Any:
    """VSC Eq. (71)."""
    p = np.asarray(p_shape_frc)
    c_p = beta_fn(1.0 / p, 1.0 + 2.0 / p) / p
    return np.pi * np.asarray(r_s) ** 2 * np.asarray(L_s) * c_p


@relation(name="FRC Ma-Xie plasma volume", tags=("frc", "geometry"), outputs="V_p")
def frc_ma_xie_plasma_volume(r_s: Any, L_s: Any, m_shape_frc: Any) -> Any:
    """Integral of VSC Eq. (72): V=pi r_s^2 L_s m/(m+1)."""
    m = np.asarray(m_shape_frc)
    return np.pi * np.asarray(r_s) ** 2 * np.asarray(L_s) * m / (m + 1.0)


@relation(name="FRC normalized enclosed volume", tags=("frc", "geometry", "default"), outputs="v_norm", dependency="generated_profile")
def frc_normalized_enclosed_volume(*, rho: Any) -> Any:
    x = np.asarray(rho, dtype=float)
    return x**2


@relation(name="FRC volume integration weight", tags=("frc", "geometry", "default"), outputs="w_V", dependency="generated_profile")
def frc_volume_integration_weight(*, rho: Any) -> Any:
    return np.asarray(rho, dtype=float).copy()
