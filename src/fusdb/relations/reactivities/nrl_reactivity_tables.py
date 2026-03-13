from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray


_CM3_TO_M3 = 1.0e-6
_TABLES_DIR = Path(__file__).with_name("tables")
_ALLOWED_INTERPOLATION_KINDS = (
    "pchip",
    "linear",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
)
_TABLE_METADATA: dict[str, dict[str, str]] = {
    "DD_total_NRL": {
        "filename": "DD_total_reactivity_NRL.txt",
        "symbolic_name": "sigmav_DD_total_NRL",
    },
    "DT_NRL": {
        "filename": "DT_reactivity_NRL.txt",
        "symbolic_name": "sigmav_DT_NRL",
    },
    "DHe3_NRL": {
        "filename": "DHe3_reactivity_NRL.txt",
        "symbolic_name": "sigmav_DHe3_NRL",
    },
    "TT_NRL": {
        "filename": "TT_reactivity_NRL.txt",
        "symbolic_name": "sigmav_TT_NRL",
    },
    "THe3_total_NRL": {
        "filename": "THe3_total_reactivity_NRL.txt",
        "symbolic_name": "sigmav_THe3_NRL",
    },
}


@dataclass(frozen=True)
class NRLReactivityTable:
    """Direct NRL Plasma Formulary reactivity table in SI units."""

    reaction_id: str
    symbolic_name: str
    temperature_grid_keV: NDArray[np.float64]
    reactivity_grid_m3_per_s: NDArray[np.float64]

    def symbolic(self, value: sp.Expr) -> sp.Expr:
        """Return a symbolic placeholder for sympy model generation."""
        return sp.Function(self.symbolic_name)(value)


def get_nrl_reactivity_table_path(reaction_id: str) -> Path:
    """Return the on-disk path for one NRL reactivity table."""
    metadata = _TABLE_METADATA[reaction_id]
    return _TABLES_DIR / metadata["filename"]


def _symbolic_placeholder(reaction_id: str, value: sp.Expr) -> sp.Expr:
    metadata = _TABLE_METADATA[reaction_id]
    return sp.Function(metadata["symbolic_name"])(value)


def _normalize_interpolation_kind(interpolation_kind: str) -> str:
    """Validate and normalize one supported NRL interpolation kind."""
    normalized = interpolation_kind.strip().lower()
    if normalized not in _ALLOWED_INTERPOLATION_KINDS:
        allowed = ", ".join(_ALLOWED_INTERPOLATION_KINDS)
        raise ValueError(
            f"Unsupported NRL interpolation_kind '{interpolation_kind}'. "
            f"Choose one of: {allowed}."
        )
    return normalized


def _read_table_rows(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read ``temperature_keV`` and ``sigmav_cm3_per_s`` numeric rows from one file."""
    temperature_keV: list[float] = []
    sigmav_cm3_per_s: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped == "//":
                continue
            parts = stripped.split()
            if len(parts) != 2:
                continue
            temperature_keV.append(float(parts[0]))
            sigmav_cm3_per_s.append(float(parts[1]))

    if not temperature_keV:
        raise ValueError(
            f"NRL reactivity table '{path.name}' contains no numeric data. "
            "Add rows as: <temperature_keV> <sigmav_cm3_per_s>."
        )

    return (
        np.asarray(temperature_keV, dtype=float),
        np.asarray(sigmav_cm3_per_s, dtype=float),
    )


@lru_cache(maxsize=None)
def load_nrl_reactivity_table(reaction_id: str) -> NRLReactivityTable:
    """Load one direct NRL reactivity table and convert from cm^3/s to m^3/s."""
    metadata = _TABLE_METADATA[reaction_id]
    temperature_keV, sigmav_cm3_per_s = _read_table_rows(get_nrl_reactivity_table_path(reaction_id))

    if np.any(temperature_keV <= 0.0):
        raise ValueError(
            f"NRL reactivity table '{metadata['filename']}' must use strictly positive temperatures in keV."
        )
    if np.any(sigmav_cm3_per_s <= 0.0):
        raise ValueError(
            f"NRL reactivity table '{metadata['filename']}' must use strictly positive reactivities in cm^3/s."
        )

    order = np.argsort(temperature_keV)
    sorted_temperature_keV = temperature_keV[order]
    sorted_sigmav_m3_per_s = sigmav_cm3_per_s[order] * _CM3_TO_M3
    unique_temperature_keV, unique_indices = np.unique(sorted_temperature_keV, return_index=True)

    return NRLReactivityTable(
        reaction_id=reaction_id,
        symbolic_name=metadata["symbolic_name"],
        temperature_grid_keV=unique_temperature_keV.astype(np.float64, copy=False),
        reactivity_grid_m3_per_s=sorted_sigmav_m3_per_s[unique_indices].astype(np.float64, copy=False),
    )


def nrl_tabulated_reactivity(
    reaction_id: str,
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Interpolate one NRL reactivity table in log-log space."""
    interpolation_kind = _normalize_interpolation_kind(interpolation_kind)
    if isinstance(ion_temp_profile, sp.Expr):
        return _symbolic_placeholder(reaction_id, ion_temp_profile)

    table = load_nrl_reactivity_table(reaction_id)
    temperatures = np.asarray(ion_temp_profile, dtype=float)
    is_scalar = temperatures.ndim == 0
    flat_temperatures = temperatures.reshape(-1)
    sigmav = np.zeros_like(flat_temperatures, dtype=float)

    positive_mask = flat_temperatures > 0.0
    if np.any(positive_mask):
        log_temperature_grid = np.log10(table.temperature_grid_keV)
        log_reactivity_grid = np.log10(table.reactivity_grid_m3_per_s)
        if interpolation_kind == "pchip":
            from scipy.interpolate import PchipInterpolator

            interpolator = PchipInterpolator(
                log_temperature_grid,
                log_reactivity_grid,
                extrapolate=False,
            )
        else:
            from scipy.interpolate import interp1d

            interpolator = interp1d(
                log_temperature_grid,
                log_reactivity_grid,
                kind=interpolation_kind,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
        interpolated = np.asarray(
            interpolator(np.log10(flat_temperatures[positive_mask])),
            dtype=float,
        )
        finite_mask = np.isfinite(interpolated)
        if np.any(finite_mask):
            sigmav_positive = np.zeros_like(interpolated, dtype=float)
            sigmav_positive[finite_mask] = np.power(10.0, interpolated[finite_mask])
            sigmav[positive_mask] = sigmav_positive

    reshaped = sigmav.reshape(temperatures.shape)
    if is_scalar:
        return float64(reshaped.item())
    return reshaped.astype(np.float64, copy=False)
