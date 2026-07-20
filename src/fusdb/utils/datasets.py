from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from fusdb.registry.constants import ATOMIC_MASS_UNIT_KG, KEV_TO_J
from fusdb.registry.dataset import (
    LoadedTable,
    load_amjuel_h2_fit,
    load_amjuel_h4_fit,
    load_dataset,
    load_table,
)
from fusdb.registry.reactivity_config import REACTIVITY_TABLES
from fusdb.registry.species_registry import SPECIES

_ALLOWED_REFERENCE_FRAMES = ("lab", "cm")
_KEV_TO_EV = 1.0e3
_CM3_S_TO_M3_S = 1.0e-6
_CM3_PER_M3 = 1.0e6
_DENSITY_SCALE_CM3 = 1.0e8


def evaluate_amjuel_h2_rate(dataset_ref: str | Path, T_edge: Any) -> Any:
    """Evaluate an AMJUEL H.2 fit and return the rate coefficient in m^3/s."""
    fit = load_amjuel_h2_fit(dataset_ref)
    T_eV = np.asarray(T_edge, dtype=float) * _KEV_TO_EV
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        log_T = np.log(T_eV)
        exponent = np.zeros(np.shape(log_T), dtype=float)
        for index, coefficient in enumerate(fit["parsed_coefficients"]):
            exponent = exponent + coefficient * (log_T**index)
        rate = np.exp(exponent) * _CM3_S_TO_M3_S
    if rate.shape == ():
        return float(rate)
    return rate


def evaluate_amjuel_h4_rate(
    dataset_ref: str | Path,
    n_e_edge: Any,
    T_edge: Any,
) -> Any:
    """Evaluate an AMJUEL H.4 fit and return the rate coefficient in m^3/s."""
    fit = load_amjuel_h4_fit(dataset_ref)
    n_cm3 = np.asarray(n_e_edge, dtype=float) / _CM3_PER_M3
    n_min, n_max = fit["parsed_density_limits_cm3"]
    n_tilde = np.clip(n_cm3, n_min, n_max) / _DENSITY_SCALE_CM3
    T_eV = np.asarray(T_edge, dtype=float) * _KEV_TO_EV
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        log_n = np.log(n_tilde)
        log_T = np.log(T_eV)
        exponent = np.zeros(
            np.broadcast_shapes(np.shape(log_n), np.shape(log_T)), dtype=float
        )
        log_n_b = np.broadcast_to(log_n, exponent.shape)
        log_T_b = np.broadcast_to(log_T, exponent.shape)
        for temperature_index in range(9):
            temperature_power = log_T_b**temperature_index
            for density_index in range(9):
                exponent = exponent + (
                    fit["coefficients"][temperature_index, density_index]
                    * temperature_power
                    * (log_n_b**density_index)
                )
        rate = np.exp(exponent) * _CM3_S_TO_M3_S
    if rate.shape == ():
        return float(rate)
    return rate


@dataclass(frozen=True)
class PreparedTable:
    """Validated and scaled table ready for downstream operations."""

    path: Path
    reaction_id: str
    metadata: dict[str, Any]
    quantities: tuple[str, ...]
    units: tuple[str, ...]
    columns: tuple[NDArray[np.float64], ...]

    def column(self, quantity: str) -> NDArray[np.float64]:
        """Return one named numeric column without exposing positional indices."""
        try:
            return self.columns[self.quantities.index(quantity)]
        except ValueError as exc:
            available = ", ".join(self.quantities)
            raise KeyError(f"Table {self.path.name!r} has no {quantity!r} column; available: {available}.") from exc


def _symbolic_name(reaction_id: str) -> str:
    if reaction_id.startswith("THe3n_"):
        suffix = reaction_id.removeprefix("THe3n_")
        return f"sigmav_THe3_np_{suffix}"
    if reaction_id.startswith("THe3D_"):
        suffix = reaction_id.removeprefix("THe3D_")
        return f"sigmav_THe3_D_{suffix}"
    if reaction_id == "THe3_total_NRL":
        return "sigmav_THe3_NRL"
    return f"sigmav_{reaction_id}"


def _reaction_id_from_table_ref(
    table_ref: str | Path,
    *,
    expected_kind: str | None = None,
) -> str:
    expected_datatype = {
        None: None,
        "cross_section": "xsection",
        "xsection": "xsection",
        "reactivity": "reactivity",
    }.get(expected_kind)
    if expected_kind is not None and expected_datatype is None:
        raise ValueError(f"Unsupported expected table kind {expected_kind!r}.")
    document = load_dataset(table_ref, expected_datatype=expected_datatype)
    subject = document.subject.replace("-total", "_total")
    return f"{subject}_{document.source.replace('-', '_')}"


def _reactant_mass_u(species: str) -> float:
    try:
        mass = SPECIES[species].isotopic_mass_u
    except KeyError as exc:
        raise ValueError(f"Unsupported reactant species '{species}'.") from exc
    return float(mass)


def _symbolic_placeholder(reaction_id: str, value: "sp.Expr") -> "sp.Expr":
    import sympy as sp

    return sp.Function(_symbolic_name(reaction_id))(value)


def _is_symbolic(value: Any) -> bool:
    """Whether ``value`` is a sympy expression, without importing sympy.

    sympy is an optional dependency (the ``datasets`` extra): if it was never
    imported, no ``sp.Expr`` instance can exist, so this is False for free.
    """
    sp = sys.modules.get("sympy")
    return sp is not None and isinstance(value, sp.Expr)


def _resolve_reference_frame(
    reference_frame: str | None,
    table_metadata: dict[str, Any],
    *,
    path: Path,
) -> str:
    if reference_frame is None:
        resolved = table_metadata.get("reference_frame", "lab")
    else:
        resolved = reference_frame
    if not isinstance(resolved, str):
        raise ValueError(f"Table '{path.name}' must use a string 'reference_frame' value.")
    normalized = resolved.strip().lower()
    if normalized not in _ALLOWED_REFERENCE_FRAMES:
        allowed = ", ".join(_ALLOWED_REFERENCE_FRAMES)
        raise ValueError(
            f"Unsupported reference_frame '{resolved}'. "
            f"Choose one of: {allowed}."
        )
    return normalized


def _reactants_from_metadata(table: LoadedTable | PreparedTable) -> tuple[str, str]:
    reactants = table.metadata.get("reactants")
    if not isinstance(reactants, dict):
        raise ValueError(
            f"Table '{table.path.name}' must define a 'reactants' mapping with projectile and target."
        )
    projectile = reactants.get("projectile")
    target = reactants.get("target")
    if not isinstance(projectile, str) or not isinstance(target, str):
        raise ValueError(
            f"Table '{table.path.name}' must define string reactants.projectile and reactants.target values."
        )
    return projectile, target


@lru_cache(maxsize=None)
def prepare_table(
    table_ref: str | Path,
    *,
    expected_kind: str | None = None,
    metadata_keys: tuple[str, ...] = (),
    quantities: tuple[str, ...],
    units: tuple[str, ...],
    scales: tuple[float, ...],
    scaled_units: tuple[str, ...] | None = None,
    positive_columns: tuple[int, ...] = (),
    sort_by: int | None = 0,
    unique_by: int | None = 0,
) -> PreparedTable:
    """Validate, scale, and optionally sort a numeric table."""
    raw_table = load_table(table_ref, metadata_keys=metadata_keys)
    if expected_kind is None:
        reaction_id = _reaction_id_from_table_ref(table_ref)
    else:
        reaction_id = _reaction_id_from_table_ref(table_ref, expected_kind=expected_kind)

    if raw_table.quantities != quantities:
        raise ValueError(
            f"Table '{raw_table.path.name}' must use columns "
            f"{', '.join(quantities)}; found {', '.join(raw_table.quantities)}."
        )
    if raw_table.units != units:
        raise ValueError(
            f"Table '{raw_table.path.name}' must use units "
            f"{', '.join(units)}; found {', '.join(raw_table.units)}."
        )
    if len(scales) != len(raw_table.columns):
        raise ValueError(
            f"Table '{raw_table.path.name}' expected {len(raw_table.columns)} scale factors; "
            f"got {len(scales)}."
        )

    columns = tuple(
        np.asarray(column * scale, dtype=float)
        for column, scale in zip(raw_table.columns, scales, strict=True)
    )
    for column_index in positive_columns:
        if np.any(columns[column_index] <= 0.0):
            raise ValueError(
                f"Table '{raw_table.path.name}' column '{quantities[column_index]}' "
                "must use strictly positive values."
            )

    if sort_by is not None:
        order = np.argsort(columns[sort_by])
        columns = tuple(np.asarray(column[order], dtype=float) for column in columns)
    if unique_by is not None:
        _, unique_indices = np.unique(columns[unique_by], return_index=True)
        columns = tuple(column[unique_indices].astype(np.float64, copy=False) for column in columns)

    return PreparedTable(
        path=raw_table.path,
        reaction_id=reaction_id,
        metadata=raw_table.metadata,
        quantities=quantities,
        units=scaled_units or units,
        columns=columns,
    )


@lru_cache(maxsize=None)
def _xsection_maxwellian_arrays(
    table_ref: str | Path, resolved_reference_frame: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return ``(energy_grid_joule, cross_section_grid_m2, reduced_mass_kg)``.

    These depend only on the table and the resolved reference frame, so they
    are built once and shared: the relations that integrate them sit inside
    least-squares residual loops, and re-interpolating the cross section onto
    the energy grid per call dominated the evaluation cost.  The returned
    arrays are shared and must not be mutated.
    """
    table = prepare_table(
        table_ref,
        expected_kind="cross_section",
        metadata_keys=("reactants", "reference_frame"),
        quantities=("energy", "cross_section"),
        units=("ev", "barn"),
        scales=(1.0e-3, 1.0e-28),
        scaled_units=("kev", "m^2"),
        sort_by=0,
        unique_by=0,
    )
    incident_energy_keV, cross_section_m2 = table.columns
    projectile, target = _reactants_from_metadata(table)
    m_projectile = _reactant_mass_u(projectile)
    m_target = _reactant_mass_u(target)
    if resolved_reference_frame == "lab":
        energy_cm_keV = incident_energy_keV * m_target / (m_projectile + m_target)
    else:
        energy_cm_keV = incident_energy_keV
    energy_grid_kev = np.logspace(
        REACTIVITY_TABLES.energy_grid_start_log10_kev,
        REACTIVITY_TABLES.energy_grid_stop_log10_kev,
        REACTIVITY_TABLES.energy_grid_num_points,
        dtype=float,
    )
    cross_section_grid_m2 = np.interp(
        energy_grid_kev,
        energy_cm_keV,
        cross_section_m2,
        left=0.0,
        right=0.0,
    )
    reduced_mass_kg = m_projectile * m_target / (m_projectile + m_target) * ATOMIC_MASS_UNIT_KG
    return energy_grid_kev * KEV_TO_J, cross_section_grid_m2, reduced_mass_kg


def reactivity_from_xsection_table(
    table_ref: str | Path,
    ion_temp_profile: "float64 | NDArray[np.float64] | sp.Expr",
    *,
    reference_frame: str | None = None,
) -> "float64 | NDArray[np.float64] | sp.Expr":
    """Return reactivity from one cross-section table file or absolute path.

    If ``reference_frame`` is omitted, the loader uses the file metadata and
    falls back to ``"lab"`` if the table does not define one. ``"lab"``
    matches the ENDF convention of projectile lab energy against a stationary target and
    converts to center-of-mass energy before integration. ``"cm"`` skips that
    conversion.
    """
    reaction_id = _reaction_id_from_table_ref(table_ref, expected_kind="cross_section")
    table = prepare_table(
        table_ref,
        expected_kind="cross_section",
        metadata_keys=("reactants", "reference_frame"),
        quantities=("energy", "cross_section"),
        units=("ev", "barn"),
        scales=(1.0e-3, 1.0e-28),
        scaled_units=("kev", "m^2"),
        sort_by=0,
        unique_by=0,
    )
    resolved_reference_frame = _resolve_reference_frame(reference_frame, table.metadata, path=table.path)
    if _is_symbolic(ion_temp_profile):
        symbolic_reaction_id = (
            reaction_id if resolved_reference_frame == "lab" else f"{reaction_id}_{resolved_reference_frame}"
        )
        return _symbolic_placeholder(symbolic_reaction_id, ion_temp_profile)

    energy_joule, cross_section_grid_m2, reduced_mass_kg = _xsection_maxwellian_arrays(
        table_ref, resolved_reference_frame
    )

    temperatures = np.asarray(ion_temp_profile, dtype=float)
    is_scalar = temperatures.ndim == 0
    flat_temperatures = temperatures.reshape(-1)
    sigmav = np.zeros_like(flat_temperatures, dtype=float)

    positive = flat_temperatures > 0.0
    if np.any(positive):
        kT = flat_temperatures[positive] * KEV_TO_J
        prefactor = np.sqrt(8.0 / (np.pi * reduced_mass_kg)) / (kT**1.5)
        integrand = (
            cross_section_grid_m2[None, :]
            * energy_joule[None, :]
            * np.exp(-energy_joule[None, :] / kT[:, None])
        )
        sigmav[positive] = prefactor * trapezoid(integrand, x=energy_joule, axis=1)

    reshaped = sigmav.reshape(temperatures.shape)
    if is_scalar:
        return float64(reshaped.item())
    return reshaped.astype(np.float64, copy=False)


@lru_cache(maxsize=None)
def _reactivity_interpolator(table_ref: str | Path, interpolation_kind: str) -> Any:
    """Return the shared log-log interpolator for one reactivity table.

    Constructing the interpolator (spline setup over the log table) costs far
    more than evaluating it, and the ``sigmav_*`` relations that use it sit
    inside least-squares residual loops, so one interpolator per
    ``(table, kind)`` is built once and reused across all evaluations.
    """
    table = prepare_table(
        table_ref,
        expected_kind="reactivity",
        quantities=("temperature", "sigmav"),
        units=("ev", "cm^3/s"),
        scales=(1.0e-3, 1.0e-6),
        scaled_units=("kev", "m^3/s"),
        positive_columns=(0, 1),
        sort_by=0,
        unique_by=0,
    )
    temperature_grid_keV, reactivity_grid_m3_per_s = table.columns
    log_temperature_grid = np.log10(temperature_grid_keV)
    log_reactivity_grid = np.log10(reactivity_grid_m3_per_s)
    if interpolation_kind == "pchip":
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(
            log_temperature_grid,
            log_reactivity_grid,
            extrapolate=False,
        )
    from scipy.interpolate import interp1d

    return interp1d(
        log_temperature_grid,
        log_reactivity_grid,
        kind=interpolation_kind,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )


def reactivity_from_reactivity_table(
    table_ref: str | Path,
    ion_temp_profile: "float64 | NDArray[np.float64] | sp.Expr",
    *,
    interpolation_kind: str = "pchip",
) -> "float64 | NDArray[np.float64] | sp.Expr":
    """Return reactivity from one direct table file or absolute path."""
    interpolation_kind = interpolation_kind.strip().lower()
    allowed_interpolation_kinds = REACTIVITY_TABLES.allowed_interpolation_kinds
    if interpolation_kind not in allowed_interpolation_kinds:
        allowed = ", ".join(allowed_interpolation_kinds)
        raise ValueError(
            f"Unsupported interpolation_kind '{interpolation_kind}'. "
            f"Choose one of: {allowed}."
        )

    reaction_id = _reaction_id_from_table_ref(table_ref, expected_kind="reactivity")
    if _is_symbolic(ion_temp_profile):
        return _symbolic_placeholder(reaction_id, ion_temp_profile)

    interpolator = _reactivity_interpolator(table_ref, interpolation_kind)
    temperatures = np.asarray(ion_temp_profile, dtype=float)
    is_scalar = temperatures.ndim == 0
    flat_temperatures = temperatures.reshape(-1)
    sigmav = np.zeros_like(flat_temperatures, dtype=float)

    positive_mask = flat_temperatures > 0.0
    if np.any(positive_mask):
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
