"""cfspopcon ``PRF`` peaked 1-D profile generators.

Adapted from cfspopcon (``cfspopcon/formulas/plasma_profiles/``); see README.md
section "Third-party Notices". The functional form is a private communication
from P. Rodriguez-Fernandez (MIT PSFC), based on TRANSP outputs, imposing:

  1. a tanh pedestal for T and n,
  2. a linear ``a/L_T`` core gradient from 0 at rho_minor=0 to the core value at
     ``rho_minor = x_a``, where ``x_a`` is set by matching the requested peaking,
  3. a flat ``a/L_T`` region from ``x_a`` to ``1 - width_ped``,
  4. the pedestal rescaled to match the requested volume average.

``x_a`` (for the core-gradient extent) and ``a/L_n`` (the density core gradient)
come from the two look-up tables ``prf_data/width.csv`` and ``prf_data/aLT.csv``
(bivariate splines over ``peaking`` and the aLT/width axes), exactly as in
cfspopcon. fusdb uses scipy/pandas (already runtime dependencies), so the LUT
handling is ported verbatim; only cfspopcon's pint unit wrapper is dropped.

The PRF shape is defined on normalized physical minor radius ``rho_minor``.
Volume normalization is performed on fusdb's common computational ``rho`` grid
with the current geometry-provided ``w_V`` when available. The reduced tokamak
mapping ``rho_minor=rho, w_V=rho`` therefore preserves the historical result,
while imported/equilibrium mappings change the shape and average consistently.

These are opt-in alternatives to the parabolic ``(1-rho^2)^alpha`` generators
(gated by ``default_relation`` on ``T_e``/``T_i``/``n_e``/``n_fuel``): the reactivity
is steeply nonlinear in ``T_i``, so the ``prf`` core shape reproduces cfspopcon's
fusion power where the parabolic shape at the same peaking over-states it.
"""

from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RectBivariateSpline

from fusdb.relation import relation
from fusdb.numerics import volume_average

# CHECK
_PRF_DATA = Path(__file__).parent / "prf_data"

# Fixed core a/L_T for the temperature profile; cfspopcon's ``evaluate_..._fits``
# default. The peaking then sets x_a (how far the linear-aLT ramp extends).
_ALT_CORE = 2.0
_WIDTH_PED = 0.05


def _load_dataframe(df_name: str) -> "pd.DataFrame":
    """Load one PRF look-up table (two header rows, two index columns).

    pandas is an optional dependency (the ``datasets`` extra), imported only
    when a PRF profile relation is actually evaluated.
    """
    import pandas as pd

    return pd.read_csv(_PRF_DATA / f"{df_name}.csv", index_col=[0, 1], header=[0, 1])


@cache
def _interpolator(df_name: str) -> RectBivariateSpline:
    """Bivariate spline over a PRF look-up table (cached per table)."""
    df = _load_dataframe(df_name)
    return RectBivariateSpline(
        [float(x[1]) for x in df.columns.values],
        [float(x[1]) for x in df.index.values],
        df.T.values,
    )


def _evaluate_profile_shape(aLT_core: float, width_axis: float, rho_minor: np.ndarray) -> np.ndarray:
    """Return the un-normalized PRF shape on normalized physical minor radius."""
    x = np.asarray(rho_minor, dtype=float)
    ix_c = int(np.argmin(np.abs(x - (1 - _WIDTH_PED))))  # extent of core
    ix_a = int(np.min([ix_c, np.argmin(np.abs(x - width_axis))]))  # extent of axis

    aLT_core = aLT_core + np.pi * 1e-8  # aLT must be non-zero

    wped_tanh = _WIDTH_PED / 1.5
    Tedge_aux = 0.5 * (1 + np.tanh((1 - x - (wped_tanh / 2)) / (wped_tanh / 2)))
    Tedge = Tedge_aux[ix_c:] / Tedge_aux[ix_c]

    Tcore_aux = np.e ** (aLT_core * (1 - _WIDTH_PED - x))
    Tcore = Tcore_aux[ix_a:ix_c]

    Taxis_aux = np.e ** (aLT_core * (-0.5 * x**2 / width_axis - 0.5 * width_axis + 1 - _WIDTH_PED))
    Taxis = Taxis_aux[:ix_a]

    return np.hstack((Taxis, Tcore, Tedge)).ravel()


@lru_cache(maxsize=64)
def _prf_unit_shape(
    peaking: float,
    nu_n: float | None,
    rho_minor_key: tuple[float, ...],
    rho_key: tuple[float, ...],
    weight_key: tuple[float, ...] | None,
) -> np.ndarray:
    """Return a PRF unit shape with unit volume average.

    ``rho_minor_key`` controls the published PRF radial shape. ``rho_key`` and
    ``weight_key`` control only the volume measure used to normalize that shape.
    ``nu_n is None`` selects a temperature profile; otherwise a density profile
    shares the temperature ``x_a`` and fits its own ``a/L_n``.
    """
    rho_minor = np.asarray(rho_minor_key, dtype=float)
    rho = np.asarray(rho_key, dtype=float)
    if rho_minor.size != rho.size or np.any(np.diff(rho_minor) <= 0.0):
        raise ValueError("rho_minor must be strictly increasing and share the rho grid")

    x_a = float(np.ravel(_interpolator("width")(_ALT_CORE, peaking))[0])
    if nu_n is None:
        aLT_core = _ALT_CORE
    else:
        aLT_core = float(np.ravel(_interpolator("aLT")(x_a, nu_n))[0])
    shape = _evaluate_profile_shape(aLT_core, x_a, rho_minor)

    weight = None if weight_key is None else np.asarray(weight_key, dtype=float)
    # Preserve the historical code path exactly for the reduced tokamak measure.
    if weight is not None and np.array_equal(weight, rho):
        weight = None
    return shape / max(float(volume_average(shape, rho, weight=weight)), 1e-300)


def _prf_profile(
    average: Any,
    peaking: Any,
    nu_n: Any,
    rho_minor: Any,
    rho: Any,
    w_V: Any = None,
) -> np.ndarray:
    """Scale a PRF shape using explicit shape-coordinate and volume semantics."""
    rho_arr = np.asarray(rho, dtype=float)
    minor_arr = np.asarray(rho_minor, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho must be a one-dimensional computational profile grid")
    if minor_arr.ndim != 1 or minor_arr.size != rho_arr.size:
        # Batched geometry mappings intentionally fall back to the relation
        # system's pointwise replay, which supplies one 1-D mapping per point.
        raise ValueError("rho_minor and rho must be one-dimensional arrays on the same grid")

    weight_key: tuple[float, ...] | None = None
    if w_V is not None:
        weight = np.asarray(w_V, dtype=float)
        if weight.ndim != 1 or weight.size != rho_arr.size:
            raise ValueError("w_V and rho must be one-dimensional arrays on the same grid")
        weight_key = tuple(float(v) for v in weight)

    peak = float(np.asarray(peaking, dtype=float).reshape(-1)[0])
    nun = None if nu_n is None else float(np.asarray(nu_n, dtype=float).reshape(-1)[0])
    unit = _prf_unit_shape(
        peak,
        nun,
        tuple(float(v) for v in minor_arr),
        tuple(float(v) for v in rho_arr),
        weight_key,
    )
    avg_arr = np.asarray(average, dtype=float)
    if avg_arr.ndim == 0:
        return float(avg_arr) * unit
    return avg_arr.reshape(avg_arr.shape[0], 1) * unit


@relation(
    name="PRF ion temperature profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="T_i",
    dependency="generated_profile",
)
def prf_ion_temperature_profile(
    T_i_avg: float,
    ion_temperature_peaking: float,
    rho_minor: Any,
    *,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Generate a cfspopcon-PRF ion-temperature profile on ``rho_minor``."""
    # CHECK
    return _prf_profile(T_i_avg, ion_temperature_peaking, None, rho_minor, rho, w_V=w_V)


@relation(
    name="PRF electron temperature profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="T_e",
    dependency="generated_profile",
)
def prf_electron_temperature_profile(
    T_e_avg: float,
    temperature_peaking: float,
    rho_minor: Any,
    *,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Generate a cfspopcon-PRF electron-temperature profile on ``rho_minor``."""
    # CHECK
    return _prf_profile(T_e_avg, temperature_peaking, None, rho_minor, rho, w_V=w_V)


@relation(
    name="PRF ion density profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="n_fuel",
    dependency="generated_profile",
)
def prf_ion_density_profile(
    n_fuel_avg: float,
    temperature_peaking: float,
    ion_density_peaking: float,
    rho_minor: Any,
    *,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Generate a cfspopcon-PRF fuel-ion density profile on ``rho_minor``."""
    # CHECK
    return _prf_profile(
        n_fuel_avg,
        temperature_peaking,
        ion_density_peaking,
        rho_minor,
        rho,
        w_V=w_V,
    )


@relation(
    name="PRF electron density profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="n_e",
    dependency="generated_profile",
)
def prf_electron_density_profile(
    n_e_avg: float,
    temperature_peaking: float,
    density_peaking: float,
    rho_minor: Any,
    *,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Generate a cfspopcon-PRF electron-density profile on ``rho_minor``."""
    # CHECK
    return _prf_profile(
        n_e_avg,
        temperature_peaking,
        density_peaking,
        rho_minor,
        rho,
        w_V=w_V,
    )
