"""Rigid-rotor field-reversed-configuration profile shapes and moments.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation


def _profile_scalar(value: Any, grid: Any) -> Any:
    arr = np.asarray(value, dtype=float)
    return arr[..., None] if arr.ndim and np.asarray(grid).ndim and arr.shape[-1] != np.asarray(grid).shape[-1] else arr


@relation(name="FRC rigid-rotor ion density profile", tags=("frc", "profile"), outputs="n_i")
def frc_rigid_rotor_ion_density(n_i_peak: Any, K_frc: Any, rho_radial: Any) -> Any:
    x = np.asarray(rho_radial, dtype=float)
    k = _profile_scalar(K_frc, x)
    n = _profile_scalar(n_i_peak, x)
    return n / np.cosh(k * (2.0 * x**2 - 1.0)) ** 2


@relation(name="FRC rigid-rotor electron density profile", tags=("frc", "profile"), outputs="n_e")
def frc_rigid_rotor_electron_density(n0: Any, K_frc: Any, rho_radial: Any) -> Any:
    x = np.asarray(rho_radial, dtype=float)
    k = _profile_scalar(K_frc, x)
    n = _profile_scalar(n0, x)
    return n / np.cosh(k * (2.0 * x**2 - 1.0)) ** 2


@relation(name="FRC rigid-rotor signed field profile", tags=("frc", "profile", "geometry"), outputs="B_signed")
def frc_rigid_rotor_signed_field(B_e: Any, K_frc: Any, rho_radial: Any) -> Any:
    x = np.asarray(rho_radial, dtype=float)
    k = _profile_scalar(K_frc, x)
    b = _profile_scalar(B_e, x)
    return b * np.tanh(k * (2.0 * x**2 - 1.0))


@relation(name="FRC G1 moment", tags=("frc", "profile"), outputs="G1_frc")
def frc_g1_moment(K_frc: Any) -> Any:
    k = np.asarray(K_frc)
    return np.tanh(k) / k


@relation(name="FRC G2 moment", tags=("frc", "profile"), outputs="G2_frc")
def frc_g2_moment(K_frc: Any) -> Any:
    k = np.asarray(K_frc)
    t = np.tanh(k)
    return (t - t**3 / 3.0) / k
