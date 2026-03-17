from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray
import yaml

from fusdb.registry import (
    ATOMIC_MASS_UNIT_KG,
    KEV_TO_J,
    REACTIVITY_ALLOWED_INTERPOLATION_KINDS,
    REACTIVITY_ENERGY_GRID_KEV,
    REACTIVITY_TABLES_DIR,
    load_allowed_species,
)


_ALLOWED_REFERENCE_FRAMES = ("lab", "cm")



@dataclass(frozen=True)
class LoadedTable:
    """Parsed numeric table plus selected metadata entries."""

    path: Path
    metadata: dict[str, Any]
    quantities: tuple[str, ...]
    units: tuple[str, ...]
    columns: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class PreparedTable:
    """Validated and scaled table ready for downstream operations."""

    path: Path
    reaction_id: str
    metadata: dict[str, Any]
    quantities: tuple[str, ...]
    units: tuple[str, ...]
    columns: tuple[NDArray[np.float64], ...]


def _normalize_quantity_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _normalize_unit_name(unit: str) -> str:
    return unit.strip().lower()


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
    path = get_table_path(table_ref)
    stem = path.stem
    if "_xsection_" in stem:
        reaction, source = stem.split("_xsection_", maxsplit=1)
        table_kind = "cross_section"
    elif "_reactivity_" in stem:
        reaction, source = stem.split("_reactivity_", maxsplit=1)
        table_kind = "reactivity"
    else:
        raise ValueError(f"Unsupported table filename '{path.name}'.")

    if expected_kind is not None and table_kind != expected_kind:
        raise ValueError(
            f"Table '{path.name}' uses table_kind '{table_kind}', expected '{expected_kind}'."
        )
    return f"{reaction}_{source.replace('-', '_')}"


def _reactant_mass_u(species: str) -> float:
    species_data = load_allowed_species()
    try:
        mass = species_data[species]["isotopic_mass_u"]
    except KeyError as exc:
        raise ValueError(f"Unsupported reactant species '{species}'.") from exc
    return float(mass)


def get_table_path(table_ref: str | Path) -> Path:
    """Resolve one table path from an explicit filename or direct path."""
    direct_path = Path(table_ref)
    if direct_path.is_absolute() or direct_path.is_file():
        return direct_path

    candidate = REACTIVITY_TABLES_DIR / direct_path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Could not resolve table '{table_ref}'.")


@lru_cache(maxsize=None)
def load_table(
    table_ref: str | Path,
    *,
    metadata_keys: tuple[str, ...] = (),
) -> LoadedTable:
    """Load one YAML-backed numeric table and return selected metadata entries."""
    path = get_table_path(table_ref)
    with path.open("r", encoding="utf-8") as handle:
        raw_document = yaml.safe_load(handle) or {}

    if not isinstance(raw_document, dict):
        raise ValueError(f"Table '{path.name}' must contain a top-level YAML mapping.")

    raw_columns = raw_document.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ValueError(
            f"Table '{path.name}' must define a non-empty 'columns' list."
        )

    quantities: list[str] = []
    units: list[str] = []
    for index, column_spec in enumerate(raw_columns):
        if not isinstance(column_spec, dict):
            raise ValueError(
                f"Table '{path.name}' column entry {index} must be a mapping with 'name' and 'unit'."
            )
        try:
            name = column_spec["name"]
            unit = column_spec["unit"]
        except KeyError as exc:
            raise ValueError(
                f"Table '{path.name}' column entry {index} must define both 'name' and 'unit'."
            ) from exc
        if not isinstance(name, str) or not isinstance(unit, str):
            raise ValueError(
                f"Table '{path.name}' column entry {index} must use string 'name' and 'unit' values."
            )
        quantities.append(_normalize_quantity_name(name))
        units.append(_normalize_unit_name(unit))

    raw_data = raw_document.get("data")
    if not isinstance(raw_data, str) or not raw_data.strip():
        raise ValueError(
            f"Table '{path.name}' must define a non-empty CSV 'data' block."
        )

    values_by_column: list[list[float]] = [[] for _ in quantities]
    row_count = 0
    reader = csv.reader(StringIO(raw_data))
    for line_number, row in enumerate(reader, start=1):
        if not row or all(not cell.strip() for cell in row):
            continue
        parts = [cell.strip() for cell in row]
        if len(parts) != len(quantities):
            raise ValueError(
                f"Table '{path.name}' CSV row {line_number} has {len(parts)} columns; "
                f"expected {len(quantities)}."
            )
        for index, part in enumerate(parts):
            values_by_column[index].append(float(part))
        row_count += 1

    if row_count == 0:
        raise ValueError(f"Table '{path.name}' must contain at least one numeric CSV row in 'data'.")

    metadata = {
        key: raw_document[key]
        for key in metadata_keys
        if key in raw_document
    }

    return LoadedTable(
        path=path,
        metadata=metadata,
        quantities=tuple(quantities),
        units=tuple(units),
        columns=tuple(np.asarray(values, dtype=float) for values in values_by_column),
    )

def _symbolic_placeholder(reaction_id: str, value: sp.Expr) -> sp.Expr:
    return sp.Function(_symbolic_name(reaction_id))(value)


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


def reactivity_from_xsection_table(
    table_ref: str | Path,
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    reference_frame: str | None = None,
) -> float64 | NDArray[np.float64] | sp.Expr:
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
    if isinstance(ion_temp_profile, sp.Expr):
        symbolic_reaction_id = (
            reaction_id if resolved_reference_frame == "lab" else f"{reaction_id}_{resolved_reference_frame}"
        )
        return _symbolic_placeholder(symbolic_reaction_id, ion_temp_profile)

    incident_energy_keV, cross_section_m2 = table.columns
    projectile, target = _reactants_from_metadata(table)
    m_projectile = _reactant_mass_u(projectile)
    m_target = _reactant_mass_u(target)
    if resolved_reference_frame == "lab":
        energy_cm_keV = incident_energy_keV * m_target / (m_projectile + m_target)
    else:
        energy_cm_keV = incident_energy_keV
    cross_section_grid_m2 = np.interp(
        REACTIVITY_ENERGY_GRID_KEV,
        energy_cm_keV,
        cross_section_m2,
        left=0.0,
        right=0.0,
    )
    reduced_mass_kg = m_projectile * m_target / (m_projectile + m_target) * ATOMIC_MASS_UNIT_KG

    temperatures = np.asarray(ion_temp_profile, dtype=float)
    is_scalar = temperatures.ndim == 0
    flat_temperatures = temperatures.reshape(-1)
    sigmav = np.zeros_like(flat_temperatures, dtype=float)

    positive = flat_temperatures > 0.0
    if np.any(positive):
        kT = flat_temperatures[positive] * KEV_TO_J
        energy_joule = REACTIVITY_ENERGY_GRID_KEV * KEV_TO_J
        prefactor = np.sqrt(8.0 / (np.pi * reduced_mass_kg)) / (kT**1.5)
        integrand = (
            cross_section_grid_m2[None, :]
            * energy_joule[None, :]
            * np.exp(-energy_joule[None, :] / kT[:, None])
        )
        sigmav[positive] = prefactor * np.trapezoid(integrand, energy_joule, axis=1)

    reshaped = sigmav.reshape(temperatures.shape)
    if is_scalar:
        return float64(reshaped.item())
    return reshaped.astype(np.float64, copy=False)


def reactivity_from_reactivity_table(
    table_ref: str | Path,
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Return reactivity from one direct table file or absolute path."""
    interpolation_kind = interpolation_kind.strip().lower()
    if interpolation_kind not in REACTIVITY_ALLOWED_INTERPOLATION_KINDS:
        allowed = ", ".join(REACTIVITY_ALLOWED_INTERPOLATION_KINDS)
        raise ValueError(
            f"Unsupported interpolation_kind '{interpolation_kind}'. "
            f"Choose one of: {allowed}."
        )

    reaction_id = _reaction_id_from_table_ref(table_ref, expected_kind="reactivity")
    if isinstance(ion_temp_profile, sp.Expr):
        return _symbolic_placeholder(reaction_id, ion_temp_profile)

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
    temperatures = np.asarray(ion_temp_profile, dtype=float)
    is_scalar = temperatures.ndim == 0
    flat_temperatures = temperatures.reshape(-1)
    sigmav = np.zeros_like(flat_temperatures, dtype=float)

    positive_mask = flat_temperatures > 0.0
    if np.any(positive_mask):
        log_temperature_grid = np.log10(temperature_grid_keV)
        log_reactivity_grid = np.log10(reactivity_grid_m3_per_s)
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
