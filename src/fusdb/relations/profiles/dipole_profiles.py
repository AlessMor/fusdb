"""Marginally stable levitated-dipole profile shapes.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC Eq. (84): n proportional to U^-1 and T proportional to U^-2/3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(name="Dipole marginal electron density profile", tags=("dipole", "profile"), outputs="n_e")
def dipole_marginal_electron_density(n0: Any, U: Any) -> Any:
    """VSC Eq. (84): n proportional to U^-1."""
    u = np.asarray(U, dtype=float)
    ratio = u / u[..., :1]
    n = np.asarray(n0, dtype=float)
    return n[..., None] / ratio if n.ndim and n.shape[-1] != ratio.shape[-1] else n / ratio


@relation(name="Dipole marginal ion density profile", tags=("dipole", "profile"), outputs="n_i")
def dipole_marginal_ion_density(n_i_peak: Any, U: Any) -> Any:
    u = np.asarray(U, dtype=float)
    ratio = u / u[..., :1]
    n = np.asarray(n_i_peak, dtype=float)
    return n[..., None] / ratio if n.ndim and n.shape[-1] != ratio.shape[-1] else n / ratio


@relation(name="Dipole marginal electron temperature profile", tags=("dipole", "profile"), outputs="T_e")
def dipole_marginal_electron_temperature(T0: Any, U: Any) -> Any:
    """VSC Eq. (84): T proportional to U^-2/3."""
    u = np.asarray(U, dtype=float)
    ratio = (u / u[..., :1]) ** (-2.0 / 3.0)
    t = np.asarray(T0, dtype=float)
    return t[..., None] * ratio if t.ndim and t.shape[-1] != ratio.shape[-1] else t * ratio


@relation(name="Dipole marginal ion temperature profile", tags=("dipole", "profile"), outputs="T_i")
def dipole_marginal_ion_temperature(T_i_peak: Any, U: Any) -> Any:
    u = np.asarray(U, dtype=float)
    ratio = (u / u[..., :1]) ** (-2.0 / 3.0)
    t = np.asarray(T_i_peak, dtype=float)
    return t[..., None] * ratio if t.ndim and t.shape[-1] != ratio.shape[-1] else t * ratio
