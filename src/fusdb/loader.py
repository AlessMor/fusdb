from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any
import warnings

import yaml

from .reactors_class import Reactor


def _section_field_map() -> tuple[dict[str, list[str]], list[str]]:
    mapping: dict[str, list[str]] = {}
    order: list[str] = []
    for field in dataclass_fields(Reactor):
        section = field.metadata.get("section")
        if section is None:
            continue
        if section not in mapping:
            mapping[section] = []
            order.append(section)
        mapping[section].append(field.name)
    return mapping, order


SECTION_FIELDS, SECTION_ORDER = _section_field_map()
REQUIRED_FIELDS = SECTION_FIELDS.get("metadata_required", [])
OPTIONAL_METADATA_FIELDS = SECTION_FIELDS.get("metadata_optional", [])
PARAMETER_SECTIONS = [
    section
    for section in SECTION_ORDER
    if section not in {"metadata_required", "metadata_optional", "artifact", "internal"}
]
PARAMETER_FIELDS = [field for section in PARAMETER_SECTIONS for field in SECTION_FIELDS[section]]
SCALAR_FIELDS = PARAMETER_FIELDS  # backward-compatible alias


def load_reactor_yaml(path: Path) -> Reactor:
    if not path.is_file():
        raise FileNotFoundError(f"reactor.yaml not found at {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"reactor.yaml at {path} must contain a mapping at the top level")

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required field(s) in {path}: {missing_list}")

    kwargs: dict[str, Any] = {field: data.get(field) for field in REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS}

    for section in PARAMETER_SECTIONS:
        section_raw = data.get(section) or {}
        if section_raw is not None and not isinstance(section_raw, dict):
            raise ValueError(f"'{section}' must be a mapping in {path}")
        for field in SECTION_FIELDS.get(section, []):
            if isinstance(section_raw, dict) and field in section_raw:
                kwargs[field] = section_raw[field]
        if section == "plasma_parameters" and isinstance(section_raw, dict):
            confinement_raw = section_raw.get("confinement_time")
            if confinement_raw is not None:
                if not isinstance(confinement_raw, dict):
                    raise ValueError(f"'plasma_parameters.confinement_time' must be a mapping in {path}")
                if "value" in confinement_raw:
                    kwargs["tau_E"] = confinement_raw.get("value")
                if "method" in confinement_raw:
                    kwargs["tau_E_method"] = confinement_raw.get("method")

    artifacts_raw = data.get("artifacts") or {}
    if artifacts_raw is not None and not isinstance(artifacts_raw, dict):
        raise ValueError(f"'artifacts' must be a mapping in {path}")
    density_profile = artifacts_raw.get("density_profile") if isinstance(artifacts_raw, dict) else None
    if density_profile is not None:
        if not isinstance(density_profile, dict):
            raise ValueError(f"'density_profile' must be a mapping in {path}")
        kwargs["density_profile_file"] = density_profile.get("file")
        kwargs["density_profile_x_axis"] = density_profile.get("x_axis")
        kwargs["density_profile_y_dataset"] = density_profile.get("y_dataset")
        kwargs["density_profile_x_unit"] = density_profile.get("x_unit")
        kwargs["density_profile_y_unit"] = density_profile.get("y_unit")

    kwargs["root_dir"] = path.parent

    density_file = kwargs.get("density_profile_file")
    if density_file:
        file_path = kwargs["root_dir"] / density_file
        if not file_path.is_file():
            warnings.warn(f"density_profile_file not found at {file_path}", UserWarning)

    return Reactor(**kwargs)


def find_reactor_dirs(root: Path) -> list[Path]:
    reactors_dir = root / "reactors"
    if not reactors_dir.is_dir():
        return []

    dirs = [
        child
        for child in reactors_dir.iterdir()
        if child.is_dir() and (child / "reactor.yaml").is_file()
    ]
    return sorted(dirs, key=lambda p: p.name)


def load_all_reactors(root: Path) -> dict[str, Reactor]:
    reactors: dict[str, Reactor] = {}
    for reactor_dir in find_reactor_dirs(root):
        reactor_path = reactor_dir / "reactor.yaml"
        reactor = load_reactor_yaml(reactor_path)
        if reactor.id in reactors:
            raise ValueError(f"Duplicate reactor id detected: {reactor.id}")
        reactors[reactor.id] = reactor
    return reactors


def reactor_table(reactors: dict[str, Reactor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reactor in sorted(reactors.values(), key=lambda r: r.id):
        rows.append(
            {
                "id": reactor.id,
                "reactor_class": reactor.reactor_class,
                "name": reactor.name,
                "reactor_configuration": reactor.reactor_configuration,
                "organization": reactor.organization,
                "country": reactor.country,
                "design_year": reactor.design_year,
                "doi": reactor.doi,
                "P_fus": reactor.P_fus,
                "R": reactor.R,
                "n_avg": reactor.n_avg,
            }
        )
    return rows
