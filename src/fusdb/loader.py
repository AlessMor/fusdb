from pathlib import Path
from typing import Any

import pint
import yaml

from .reactors_class import Reactor
from .reactor_util import (
    ALLOWED_CONFINEMENT_MODES,
    ALLOWED_REACTOR_FAMILIES,
    ALLOWED_VARIABLES,
    ARTIFACT_FIELDS,
    OPTIONAL_METADATA_FIELDS,
    REQUIRED_FIELDS,
    RESERVED_KEYS,
    configuration_tags,
    default_parameters_for_tags,
    normalize_allowed,
    variable_aliases,
)

_UNIT_REGISTRY = pint.UnitRegistry()
def load_reactor_yaml(path: Path | str) -> Reactor:
    """Load a reactor YAML file and return a fully solved Reactor instance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reactor.yaml not found at {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"reactor.yaml at {path} must contain a mapping at the top level")

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required field(s) in {path}: {missing_list}")

    kwargs: dict[str, Any] = {field: data.get(field) for field in REQUIRED_FIELDS}
    for field in OPTIONAL_METADATA_FIELDS:
        if field in data:
            kwargs[field] = data.get(field)
    for field in ARTIFACT_FIELDS:
        if field in data:
            kwargs[field] = data.get(field)

    parameters: dict[str, Any] = {}
    tolerances: dict[str, float] = {}
    parameter_methods: dict[str, str] = {}
    param_sources: dict[str, str] = {}
    explicit_parameters: set[str] = set()
    aliases = variable_aliases()

    for key, value in data.items():
        if key in RESERVED_KEYS:
            continue
        canonical = aliases.get(key, key)
        if canonical in param_sources:
            other = param_sources[canonical]
            raise ValueError(
                f"Duplicate parameter {canonical!r} defined via {other!r} and {key!r} in {path}"
            )
        param_sources[canonical] = key
        # Parse a scalar or value/tolerance/method mapping from the YAML entry.
        param_value = value
        param_tol = None
        param_method = None
        param_unit = None
        if isinstance(value, dict):
            allowed_keys = {"value", "tol", "tolerance", "method", "unit"}
            if not set(value.keys()).issubset(allowed_keys):
                raise ValueError(
                    f"Parameter {canonical!r} must be a scalar or a mapping with 'value'/'tol'/'method'/'unit' keys"
                )
            param_value = value.get("value")
            tol_raw = value.get("tol", value.get("tolerance"))
            param_tol = None if tol_raw is None else float(tol_raw)
            method_raw = value.get("method")
            param_method = None if method_raw is None else str(method_raw)
            unit_raw = value.get("unit")
            param_unit = None if unit_raw is None else str(unit_raw)
        if param_unit is not None:
            if param_value is None:
                raise ValueError(f"Parameter {canonical!r} unit provided without a value in {path}")
            meta = ALLOWED_VARIABLES.get(canonical)
            if not isinstance(meta, dict):
                raise ValueError(f"Parameter {canonical!r} has no default_unit for conversion in {path}")
            default_unit = meta.get("default_unit")
            if not default_unit:
                raise ValueError(f"Parameter {canonical!r} has no default_unit for conversion in {path}")
            try:
                magnitude = float(param_value)
                quantity = _UNIT_REGISTRY.Quantity(magnitude, param_unit)
                param_value = quantity.to(str(default_unit)).magnitude
            except Exception as exc:
                raise ValueError(
                    f"Failed to convert {canonical!r} from {param_unit!r} to {default_unit!r} in {path}"
                ) from exc
        parameters[canonical] = param_value
        explicit_parameters.add(canonical)
        if param_tol is not None:
            tolerances[canonical] = param_tol
        if param_method is not None:
            parameter_methods[canonical] = param_method

    reactor_family = data.get("reactor_family")
    confinement_mode = data.get("confinement_mode")
    normalized_family = reactor_family
    if reactor_family is not None and ALLOWED_REACTOR_FAMILIES:
        normalized_family = normalize_allowed(
            reactor_family, ALLOWED_REACTOR_FAMILIES, field_name="reactor_family"
        )
    normalized_mode = confinement_mode
    if confinement_mode is not None and ALLOWED_CONFINEMENT_MODES:
        normalized_mode = normalize_allowed(
            confinement_mode, ALLOWED_CONFINEMENT_MODES, field_name="confinement_mode"
        )
    default_tags = set(configuration_tags(data.get("reactor_configuration")))
    if normalized_family:
        default_tags.add(str(normalized_family))
    if normalized_mode:
        default_tags.add(str(normalized_mode))
    parameter_defaults: dict[str, Any] = {
        name: default
        for name, default in default_parameters_for_tags(default_tags).items()
        if name not in parameters
    }

    kwargs["parameters"] = parameters
    kwargs["parameter_tolerances"] = tolerances
    kwargs["parameter_methods"] = parameter_methods
    kwargs["parameter_defaults"] = parameter_defaults
    kwargs["explicit_parameters"] = explicit_parameters
    kwargs["root_dir"] = path.parent

    return Reactor(**kwargs)


def find_reactor_dirs(root: Path) -> list[Path]:
    """Return reactor directories containing a reactor.yaml under the given root."""
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
    """Load all reactors under root/reactors keyed by reactor id."""
    reactors: dict[str, Reactor] = {}
    for reactor_dir in find_reactor_dirs(root):
        reactor_path = reactor_dir / "reactor.yaml"
        reactor = load_reactor_yaml(reactor_path)
        if reactor.id in reactors:
            raise ValueError(f"Duplicate reactor id detected: {reactor.id}")
        reactors[reactor.id] = reactor
    return reactors


def reactor_table(reactors: dict[str, Reactor]) -> list[dict[str, Any]]:
    """Build a summary table of selected metadata and parameters."""
    rows: list[dict[str, Any]] = []
    for reactor in sorted(reactors.values(), key=lambda r: r.id):
        rows.append(
            {
                "id": reactor.id,
                "reactor_family": reactor.reactor_family,
                "name": reactor.name,
                "reactor_configuration": reactor.reactor_configuration,
                "organization": reactor.organization,
                "country": reactor.country,
                "design_year": reactor.design_year,
                "doi": reactor.doi,
                "P_fus": reactor.parameters.get("P_fus"),
                "R": reactor.parameters.get("R"),
                "n_avg": reactor.parameters.get("n_avg"),
            }
        )
    return rows
