"""Registry and parsers for packaged scientific datasets.

Every packaged YAML resource has a common four-field envelope:
``schema_version``, ``datatype``, ``source`` and ``subject``.  The remaining
fields are deliberately datatype-specific; cross sections, direct
reactivities, polynomial fits and cooling curves do not share one artificial
payload shape.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from io import StringIO
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray
import yaml


SCHEMA_VERSION = 1
SUPPORTED_DATATYPES = frozenset(
    {"xsection", "reactivity", "polynomialfit", "coolingcurve", "meancharge", "meansquarecharge"}
)


@dataclass(frozen=True, slots=True)
class DatasetDocument:
    """One validated dataset document and its packaged resource path."""

    dataset_id: str
    path: Path
    datatype: str
    source: str
    subject: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedTable:
    """Parsed numeric columns from a tabular dataset."""

    dataset_id: str
    path: Path
    metadata: dict[str, Any]
    quantities: tuple[str, ...]
    units: tuple[str, ...]
    columns: tuple[NDArray[np.float64], ...]


def dataset_filename(datatype: str, source: str, subject: str) -> str:
    """Return the canonical ``datatype_source_subject.yaml`` filename."""
    return f"{datatype}_{source}_{subject}.yaml"


class DatasetRegistry:
    """Index packaged datasets by stable filename stem."""

    def __init__(self, resource_root: Path | None = None) -> None:
        if resource_root is None:
            resource_root = Path(str(files(__package__)))
        self.resource_root = Path(resource_root)
        paths = sorted(self.resource_root.rglob("*.yaml"))
        index: dict[str, Path] = {}
        for path in paths:
            dataset_id = path.stem
            if dataset_id in index:
                raise ValueError(
                    f"Duplicate dataset id {dataset_id!r}: {index[dataset_id]} and {path}."
                )
            index[dataset_id] = path
        self._paths = index

    def __contains__(self, dataset_id: object) -> bool:
        return isinstance(dataset_id, str) and dataset_id.removesuffix(".yaml") in self._paths

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def resolve(self, dataset_ref: str | Path) -> Path:
        """Resolve a dataset ID, canonical filename, or explicit filesystem path."""
        direct = Path(dataset_ref)
        if direct.is_absolute() or direct.is_file():
            if direct.is_file():
                return direct.resolve()
            raise FileNotFoundError(f"Dataset path does not exist: {direct}.")

        dataset_id = str(dataset_ref).removesuffix(".yaml")
        try:
            return self._paths[dataset_id]
        except KeyError as exc:
            raise FileNotFoundError(f"Unknown dataset {dataset_ref!r}.") from exc

    def load(
        self,
        dataset_ref: str | Path,
        *,
        expected_datatype: str | None = None,
    ) -> DatasetDocument:
        return load_dataset(dataset_ref, expected_datatype=expected_datatype, registry=self)


DATASETS = DatasetRegistry()


def _validate_document(path: Path, raw: Any, *, expected_datatype: str | None) -> DatasetDocument:
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset {path.name!r} must contain a top-level YAML mapping.")

    missing = [
        key for key in ("schema_version", "datatype", "source", "subject") if key not in raw
    ]
    if missing:
        raise ValueError(f"Dataset {path.name!r} is missing common fields: {', '.join(missing)}.")

    version = raw["schema_version"]
    datatype = str(raw["datatype"]).strip()
    source = str(raw["source"]).strip()
    subject = str(raw["subject"]).strip()
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Dataset {path.name!r} uses schema_version {version!r}; expected {SCHEMA_VERSION}."
        )
    if datatype not in SUPPORTED_DATATYPES:
        raise ValueError(f"Dataset {path.name!r} uses unsupported datatype {datatype!r}.")
    if expected_datatype is not None and datatype != expected_datatype:
        raise ValueError(
            f"Dataset {path.name!r} uses datatype {datatype!r}; expected {expected_datatype!r}."
        )
    if not source or not subject:
        raise ValueError(f"Dataset {path.name!r} must use non-empty source and subject values.")

    expected_name = dataset_filename(datatype, source, subject)
    if path.name != expected_name:
        raise ValueError(
            f"Dataset {path.name!r} does not follow its declared canonical name {expected_name!r}."
        )

    if datatype in {"xsection", "reactivity"}:
        if not isinstance(raw.get("columns"), list) or not raw["columns"]:
            raise ValueError(f"Tabular dataset {path.name!r} must define non-empty columns.")
        if not isinstance(raw.get("data"), str) or not raw["data"].strip():
            raise ValueError(f"Tabular dataset {path.name!r} must define a CSV data block.")
    elif datatype == "polynomialfit":
        if not any(key in raw for key in ("coefficients", "coefficient_blocks", "radc")):
            raise ValueError(f"Polynomial-fit dataset {path.name!r} has no coefficients.")
    elif datatype == "coolingcurve":
        for key in ("temperature_keV", "Lz_Wm3"):
            if key not in raw:
                raise ValueError(f"Cooling-curve dataset {path.name!r} is missing {key!r}.")
    elif datatype == "meancharge":
        for key in ("temperature_keV", "electron_density_m3", "mean_charge"):
            if key not in raw:
                raise ValueError(f"Mean-charge dataset {path.name!r} is missing {key!r}.")
    elif datatype == "meansquarecharge":
        for key in ("temperature_keV", "electron_density_m3", "mean_square_charge"):
            if key not in raw:
                raise ValueError(f"Mean-square-charge dataset {path.name!r} is missing {key!r}.")

    return DatasetDocument(
        dataset_id=path.stem,
        path=path,
        datatype=datatype,
        source=source,
        subject=subject,
        data=raw,
    )


@lru_cache(maxsize=None)
def _load_dataset_cached(
    path: Path,
    expected_datatype: str | None,
) -> DatasetDocument:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _validate_document(path, raw, expected_datatype=expected_datatype)


def load_dataset(
    dataset_ref: str | Path,
    *,
    expected_datatype: str | None = None,
    registry: DatasetRegistry = DATASETS,
) -> DatasetDocument:
    """Resolve, parse and validate one scientific dataset."""
    path = registry.resolve(dataset_ref)
    return _load_dataset_cached(path, expected_datatype)


def parse_fortran_float(value: Any) -> float:
    """Parse a scientific number that may use Fortran ``D`` exponents."""
    text = str(value).strip().replace("D", "E").replace("d", "E")
    text = text.replace("E ", "E+").replace("e ", "E+")
    return float(text)


@lru_cache(maxsize=None)
def load_amjuel_h2_fit(dataset_ref: str | Path) -> dict[str, Any]:
    """Parse one AMJUEL H.2 polynomial-fit resource."""
    document = load_dataset(dataset_ref, expected_datatype="polynomialfit")
    coefficients = document.data.get("coefficients", ())
    if len(coefficients) != 9:
        raise ValueError(f"Expected 9 AMJUEL H.2 coefficients in {document.path}.")
    parsed = np.array([parse_fortran_float(value) for value in coefficients], dtype=float)
    return {**document.data, "path": document.path, "parsed_coefficients": parsed}


@lru_cache(maxsize=None)
def load_amjuel_h4_fit(dataset_ref: str | Path) -> dict[str, Any]:
    """Parse one AMJUEL H.4 coefficient-block resource."""
    document = load_dataset(dataset_ref, expected_datatype="polynomialfit")
    blocks = document.data.get("coefficient_blocks", ())
    coefficients = np.zeros((9, 9), dtype=float)
    for block in blocks:
        density_indices = [int(item) for item in block["density_indices"]]
        for row in block["rows"]:
            temperature_index = int(row["temperature_index"])
            values = row["coefficients"]
            if len(values) != len(density_indices):
                raise ValueError(f"Bad AMJUEL row width in {document.path}.")
            for density_index, value in zip(density_indices, values):
                coefficients[temperature_index, density_index] = parse_fortran_float(value)

    density_limits = document.data.get("density_limits", {})
    parsed_density_limits = (
        parse_fortran_float(density_limits["min_cm3"]),
        parse_fortran_float(density_limits["max_cm3"]),
    )
    return {
        **document.data,
        "path": document.path,
        "coefficients": coefficients,
        "parsed_density_limits_cm3": parsed_density_limits,
    }


@lru_cache(maxsize=None)
def load_table(
    dataset_ref: str | Path,
    *,
    metadata_keys: tuple[str, ...] = (),
) -> LoadedTable:
    """Parse numeric columns from an xsection or reactivity YAML dataset."""
    document = load_dataset(dataset_ref)
    if document.datatype not in {"xsection", "reactivity"}:
        raise ValueError(f"Dataset {document.dataset_id!r} is not tabular.")

    quantities: list[str] = []
    units: list[str] = []
    for index, column_spec in enumerate(document.data["columns"]):
        if not isinstance(column_spec, dict):
            raise ValueError(
                f"Dataset {document.path.name!r} column {index} must be a mapping."
            )
        name = column_spec.get("name")
        unit = column_spec.get("unit")
        if not isinstance(name, str) or not isinstance(unit, str):
            raise ValueError(
                f"Dataset {document.path.name!r} column {index} needs string name and unit."
            )
        quantities.append(name.strip().lower().replace("-", "_"))
        units.append(unit.strip().lower())

    values_by_column: list[list[float]] = [[] for _ in quantities]
    row_count = 0
    for line_number, row in enumerate(csv.reader(StringIO(document.data["data"])), start=1):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(quantities):
            raise ValueError(
                f"Dataset {document.path.name!r} CSV row {line_number} has {len(row)} "
                f"columns; expected {len(quantities)}."
            )
        for index, value in enumerate(row):
            values_by_column[index].append(float(value.strip()))
        row_count += 1
    if not row_count:
        raise ValueError(f"Dataset {document.path.name!r} has no numeric rows.")

    metadata = {key: document.data[key] for key in metadata_keys if key in document.data}
    return LoadedTable(
        dataset_id=document.dataset_id,
        path=document.path,
        metadata=metadata,
        quantities=tuple(quantities),
        units=tuple(units),
        columns=tuple(np.asarray(values, dtype=float) for values in values_by_column),
    )


__all__ = [
    "DATASETS",
    "SCHEMA_VERSION",
    "SUPPORTED_DATATYPES",
    "DatasetDocument",
    "DatasetRegistry",
    "LoadedTable",
    "dataset_filename",
    "load_amjuel_h2_fit",
    "load_amjuel_h4_fit",
    "load_dataset",
    "load_table",
    "parse_fortran_float",
]
