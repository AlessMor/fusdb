"""Confinement-mode-aware pedestal profile relations.

The PROCESS relations port the HELIOS pedestal parameterisation used by
``process.models.physics.profiles``.  The FUSE alternatives port the H-mode
profile shape that FUSE calls through its IMAS.jl dependency.  Relations are
kept source-distinct and selectable by name; the PROCESS variants carry
``confinement_mode_profile_default`` so :class:`fusdb.reactor.Reactor` can
prefer them over a generic parabolic shape when the required pedestal data are
supplied.

Adapted from PROCESS and FUSE/IMAS.jl; see README.md section
"Third-party Notices".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import beta as beta_fn

from fusdb.relation import relation


_MODE_PROFILE = ("plasma", "profile", "profile_shape", "confinement_mode_profile_default")
_FUSE_PROFILE = ("plasma", "profile", "profile_shape")


def _column(value: Any) -> np.ndarray:
    """Return a scalar unchanged or a batched value as an ``(N, 1)`` column."""
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 0 else arr.reshape(-1, 1)


def _finish_profile(profile: np.ndarray, *values: Any) -> np.ndarray:
    """Drop the synthetic batch axis when every physical input was scalar."""
    if all(np.asarray(value).ndim == 0 for value in values):
        return np.asarray(profile, dtype=float).reshape(-1)
    return np.asarray(profile, dtype=float)


def _process_temperature_profile(
    average: Any,
    pedestal: Any,
    separatrix: Any,
    pedestal_radius: Any,
    alpha: Any,
    beta_exponent: Any,
    rho: Any,
) -> np.ndarray:
    """PROCESS/HELIOS pedestal temperature profile normalized by its average."""
    x = np.asarray(rho, dtype=float)
    if x.ndim != 1:
        raise ValueError("rho must be a one-dimensional profile grid")
    avg = _column(average)
    ped = _column(pedestal)
    sep = _column(separatrix)
    rped = _column(pedestal_radius)
    alphat = _column(alpha)
    tbeta = _column(beta_exponent)

    core = ped + (
        tbeta
        * (
            3.0 * avg
            + sep * (-2.0 + rped + rped**2)
            - ped * (1.0 + rped + rped**2)
        )
        / (6.0 * rped**2 * beta_fn(1.0 + alphat, 2.0 / tbeta))
    )
    inside = x <= rped
    core_part = ped + (core - ped) * np.maximum(1.0 - (x / rped) ** tbeta, 0.0) ** alphat
    edge_part = sep + (ped - sep) * (1.0 - x) / np.maximum(1.0 - rped, 1e-12)
    profile = np.where(inside, core_part, edge_part)
    return _finish_profile(profile, average, pedestal, separatrix, pedestal_radius, alpha, beta_exponent)


def _process_density_profile(
    average: Any,
    pedestal: Any,
    separatrix: Any,
    pedestal_radius: Any,
    alpha: Any,
    rho: Any,
) -> np.ndarray:
    """PROCESS/HELIOS pedestal density profile normalized by its average."""
    x = np.asarray(rho, dtype=float)
    if x.ndim != 1:
        raise ValueError("rho must be a one-dimensional profile grid")
    avg = _column(average)
    ped = _column(pedestal)
    sep = _column(separatrix)
    rped = _column(pedestal_radius)
    alphan = _column(alpha)

    core = (
        3.0 * avg * (1.0 + alphan)
        + sep * (1.0 + alphan) * (-2.0 + rped + rped**2)
        - ped * ((1.0 + alphan) * (1.0 + rped) + (alphan - 2.0) * rped**2)
    ) / (3.0 * rped**2)
    inside = x <= rped
    core_part = ped + (core - ped) * np.maximum(1.0 - (x / rped) ** 2, 0.0) ** alphan
    edge_part = sep + (ped - sep) * (1.0 - x) / np.maximum(1.0 - rped, 1e-12)
    profile = np.where(inside, core_part, edge_part)
    return _finish_profile(profile, average, pedestal, separatrix, pedestal_radius, alpha)


@relation(
    name="PROCESS pedestal electron temperature profile",
    tags=(*_MODE_PROFILE, "process", "h_mode", "i_mode"),
    outputs="T_e",
    dependency="generated_profile",
)
def process_pedestal_electron_temperature_profile(
    T_e_avg: Any,
    temp_plasma_pedestal_kev: Any,
    T_sep: Any,
    radius_plasma_pedestal_temp_norm: Any,
    alphat: Any,
    tbeta: Any,
    rho: Any,
) -> np.ndarray:
    """Generate the PROCESS/HELIOS pedestal electron-temperature profile.

    H-mode and I-mode both carry a temperature pedestal. Adapted from PROCESS;
    see README.md section "Third-party Notices".
    """
    return _process_temperature_profile(
        T_e_avg,
        temp_plasma_pedestal_kev,
        T_sep,
        radius_plasma_pedestal_temp_norm,
        alphat,
        tbeta,
        rho,
    )


@relation(
    name="PROCESS pedestal ion temperature profile",
    tags=(*_MODE_PROFILE, "process", "h_mode", "i_mode"),
    outputs="T_i",
    dependency="generated_profile",
)
def process_pedestal_ion_temperature_profile(
    T_i_avg: Any,
    T_e_avg: Any,
    temp_plasma_pedestal_kev: Any,
    T_sep: Any,
    radius_plasma_pedestal_temp_norm: Any,
    alphat: Any,
    tbeta: Any,
    rho: Any,
) -> np.ndarray:
    """PROCESS pedestal shape for ions, scaled by the volume-average Ti/Te ratio."""
    ratio = np.asarray(T_i_avg, dtype=float) / np.maximum(np.asarray(T_e_avg, dtype=float), 1e-300)
    return _process_temperature_profile(
        T_i_avg,
        np.asarray(temp_plasma_pedestal_kev) * ratio,
        np.asarray(T_sep) * ratio,
        radius_plasma_pedestal_temp_norm,
        alphat,
        tbeta,
        rho,
    )


@relation(
    name="PROCESS pedestal electron density profile",
    tags=(*_MODE_PROFILE, "process", "h_mode"),
    outputs="n_e",
    dependency="generated_profile",
)
def process_pedestal_electron_density_profile(
    n_e_avg: Any,
    nd_plasma_pedestal_electron: Any,
    n_sep: Any,
    radius_plasma_pedestal_density_norm: Any,
    alphan: Any,
    rho: Any,
) -> np.ndarray:
    """Generate the PROCESS/HELIOS H-mode electron-density pedestal profile."""
    return _process_density_profile(
        n_e_avg,
        nd_plasma_pedestal_electron,
        n_sep,
        radius_plasma_pedestal_density_norm,
        alphan,
        rho,
    )


@relation(
    name="PROCESS pedestal fuel-ion density profile",
    tags=(*_MODE_PROFILE, "process", "h_mode"),
    outputs="n_fuel",
    dependency="generated_profile",
)
def process_pedestal_fuel_ion_density_profile(
    n_fuel_avg: Any,
    n_e_avg: Any,
    nd_plasma_pedestal_electron: Any,
    n_sep: Any,
    radius_plasma_pedestal_density_norm: Any,
    alphan: Any,
    rho: Any,
) -> np.ndarray:
    """PROCESS H-mode density shape scaled to the fuel-ion average."""
    ratio = np.asarray(n_fuel_avg, dtype=float) / np.maximum(np.asarray(n_e_avg, dtype=float), 1e-300)
    return _process_density_profile(
        n_fuel_avg,
        np.asarray(nd_plasma_pedestal_electron) * ratio,
        np.asarray(n_sep) * ratio,
        radius_plasma_pedestal_density_norm,
        alphan,
        rho,
    )


def _fuse_hmode_profile(
    edge: Any,
    pedestal: Any,
    core: Any,
    exponent_inner: Any,
    exponent_outer: Any,
    width: Any,
    rho: Any,
) -> np.ndarray:
    """FUSE/IMAS ``Hmode_profiles`` on the fusdb radial grid."""
    x = np.asarray(rho, dtype=float)
    if x.ndim != 1:
        raise ValueError("rho must be a one-dimensional profile grid")
    edge_v = _column(edge)
    ped = _column(pedestal)
    core_v = _column(core)
    expin = _column(exponent_inner)
    expout = _column(exponent_outer)
    full_width = _column(width)

    half_width = 0.5 * full_width
    x_half = 1.0 - half_width
    pconst = 1.0 - np.tanh((1.0 - x_half) / half_width)
    amplitude = 2.0 * (ped - edge_v) / (1.0 + np.tanh(1.0) - pconst)
    core_tanh = 0.5 * amplitude * (1.0 - np.tanh(-x_half / half_width) - pconst) + edge_v
    profile = 0.5 * amplitude * (1.0 - np.tanh((x - x_half) / half_width) - pconst) + edge_v
    x_ped = x_half - half_width
    x_to_ped = x / x_ped
    core_shape = np.maximum(1.0 - np.maximum(x_to_ped, 0.0) ** expin, 0.0) ** expout
    profile = profile + np.where(x_to_ped < 1.0, (core_v - core_tanh) * core_shape, 0.0)
    return _finish_profile(profile, edge, pedestal, core, exponent_inner, exponent_outer, width)


@relation(
    name="FUSE IMAS H-mode electron temperature profile",
    tags=(*_FUSE_PROFILE, "h_mode", "i_mode"),
    outputs="T_e",
    dependency="generated_profile",
)
def fuse_imas_hmode_electron_temperature_profile(
    T_sep: Any,
    temp_plasma_pedestal_kev: Any,
    T0: Any,
    alphat: Any,
    pedestal_width: Any,
    rho: Any,
) -> np.ndarray:
    """FUSE/IMAS H-mode temperature shape; also applicable to I-mode."""
    return _fuse_hmode_profile(
        T_sep, temp_plasma_pedestal_kev, T0, alphat, alphat, pedestal_width, rho
    )


@relation(
    name="FUSE IMAS H-mode electron density profile",
    tags=(*_FUSE_PROFILE, "h_mode"),
    outputs="n_e",
    dependency="generated_profile",
)
def fuse_imas_hmode_electron_density_profile(
    n_sep: Any,
    nd_plasma_pedestal_electron: Any,
    n0: Any,
    alphan: Any,
    pedestal_width: Any,
    rho: Any,
) -> np.ndarray:
    """FUSE/IMAS H-mode density shape; deliberately excluded from I-mode."""
    return _fuse_hmode_profile(
        n_sep, nd_plasma_pedestal_electron, n0, alphan, alphan, pedestal_width, rho
    )
