"""Registry module for allowed variables, tags, and constants."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import warnings
from ..utils import within_tolerance, load_yaml, normalize_tag, normalize_tags_to_tuple, normalize_country
from ..variable_class import Variable

# Registry paths
REGISTRY_PATH = Path(__file__).resolve().parent
TAGS_PATH = REGISTRY_PATH / "allowed_tags.yaml"
VARIABLES_PATH = REGISTRY_PATH / "allowed_variables.yaml"
SPECIES_PATH = REGISTRY_PATH / "allowed_species.yaml"
SOLVER_DEFAULTS_PATH = REGISTRY_PATH / "solver_defaults.yaml"

# Private caches
_ALLOWED_VARIABLES: dict[str, dict] | None = None
_ALIASES: dict[str, str] | None = None
_DEFAULT_UNITS: dict[str, str] | None = None
_ALLOWED_TAGS: dict[str, Any] | None = None
_CONSTANTS: dict[str, float] | None = None
_SOLVER_DEFAULTS: dict[str, Any] | None = None


def load_constants() -> dict[str, float]:
    """Load physical constants from YAML. Args: none. Returns: dict."""
    global _CONSTANTS
    if _CONSTANTS is None:
        _CONSTANTS = load_yaml(REGISTRY_PATH / "constants.yaml")
    return _CONSTANTS


def load_solver_defaults() -> dict[str, Any]:
    """Load solver defaults. Args: none. Returns: dict."""
    global _SOLVER_DEFAULTS
    if _SOLVER_DEFAULTS is None:
        _SOLVER_DEFAULTS = load_yaml(SOLVER_DEFAULTS_PATH)
    return _SOLVER_DEFAULTS


def load_allowed_variables() -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    """Load allowed variables. Args: none. Returns: (data, alias_map, default_units)."""
    global _ALLOWED_VARIABLES, _ALIASES, _DEFAULT_UNITS
    if _ALLOWED_VARIABLES is not None:
        return _ALLOWED_VARIABLES, _ALIASES or {}, _DEFAULT_UNITS or {}
    data = load_yaml(REGISTRY_PATH / "allowed_variables.yaml")
    alias_map: dict[str, str] = {}
    default_units: dict[str, str] = {}
    for name, spec in data.items():
        default_units[name] = spec.get("default_unit")
        for alias in spec.get("aliases", []) or []:
            alias_map[alias] = name
    _ALLOWED_VARIABLES, _ALIASES, _DEFAULT_UNITS = data, alias_map, default_units
    return data, alias_map, default_units


def allowed_variable_constraints(name: str) -> tuple[str, ...]:
    """Return constraints for an allowed variable. Args: name. Returns: tuple[str,...]."""
    data, _, _ = load_allowed_variables()
    return tuple((data.get(name, {}) or {}).get("constraints") or ())


def _load_allowed_tags() -> dict[str, Any]:
    """Load allowed tags metadata. Args: none. Returns: dict."""
    global _ALLOWED_TAGS
    if _ALLOWED_TAGS is None:
        tags = load_yaml(REGISTRY_PATH / "allowed_tags.yaml")
        if "solving_order" in tags:
            tags["solving_order"] = {normalize_tag(k): v for k, v in (tags.get("solving_order") or {}).items()}
        for key in ("confinement_modes", "reactor_families", "reactor_configurations"):
            if key in tags:
                tags[key] = list(normalize_tags_to_tuple(tags.get(key) or []))
        _ALLOWED_TAGS = tags
    return _ALLOWED_TAGS


# Public constants loaded from tags
ALLOWED_VARIABLES = tuple(load_allowed_variables()[0].keys())
ALLOWED_SOLVING_ORDER = tuple(_load_allowed_tags().get("solving_order", {}).keys())
ALLOWED_CONFINEMENT_MODES = tuple(_load_allowed_tags().get("confinement_modes", []))
ALLOWED_REACTOR_FAMILIES = tuple(_load_allowed_tags().get("reactor_families", []))
ALLOWED_REACTOR_CONFIGURATIONS = tuple(_load_allowed_tags().get("reactor_configurations", []))
OPTIONAL_METADATA_FIELDS = tuple(_load_allowed_tags().get("optional_metadata_fields", []))
REQUIRED_FIELDS = tuple(_load_allowed_tags().get("required_metadata_fields", []))
RESERVED_KEYS = ("metadata", "tags", "solver_tags", "variables")

# Load constants as module-level attributes
_constants_dict = load_constants()
for _const_name, _const_value in _constants_dict.items():
    globals()[_const_name] = _const_value


def canonical_variable_name(name: str) -> str:
    """Return canonical variable name (resolve aliases). Args: name. Returns: canonical name."""
    if _ALIASES is None:
        load_allowed_variables()
    return _ALIASES.get(name, name)


def parse_variables(variables: dict[str, Any] | None) -> dict[str, Variable]:
    """Parse variables from YAML. Args: variables. Returns: variables dict."""
    _, _alias_map, default_units = load_allowed_variables()
    parsed: dict[str, Variable] = {}
    raw_vars = variables or {}
    fractions = raw_vars.get("fractions") if isinstance(raw_vars, dict) else None
    if isinstance(fractions, dict):
        base = {k: v for k, v in raw_vars.items() if not str(k).startswith("f_") and k != "fractions"}
        for species, frac in fractions.items():
            name = canonical_variable_name(f"f_{species}")
            base[name] = {"value": frac}
        raw_vars = base
    def _normalize_entry(entry: object) -> dict[str, object]:
        if isinstance(entry, dict):
            return {
                "value": entry.get("value"),
                "unit": entry.get("unit"),
                "method": entry.get("method"),
                "rel_tol": entry.get("rel_tol"),
                "abs_tol": entry.get("abs_tol"),
                "fixed": bool(entry.get("fixed", False)),
            }
        return {"value": entry, "unit": None, "method": None, "rel_tol": None, "abs_tol": None, "fixed": False}
    grouped: dict[str, list[tuple[str, object]]] = {}
    if isinstance(raw_vars, dict):
        for raw_name, entry in raw_vars.items():
            if raw_name == "fractions":
                continue
            grouped.setdefault(canonical_variable_name(raw_name), []).append((raw_name, entry))
    for name, entries in grouped.items():
        merged: dict[str, object] = {"value": None, "unit": None, "method": None, "rel_tol": None, "abs_tol": None, "fixed": False}
        value_source = unit_source = method_source = None
        conflicts: list[str] = []
        for raw_name, entry in entries:
            item = _normalize_entry(entry)
            value = item.get("value")
            if value is not None:
                if merged["value"] is None:
                    merged["value"] = value
                    value_source = raw_name
                elif merged["value"] != value:
                    # Check numeric tolerance if both are numeric
                    values_conflict = True
                    if merged["value"] is not None and value is not None:
                        try:
                            float(merged["value"])
                            float(value)
                            values_conflict = not within_tolerance(merged["value"], value)
                        except Exception:
                            pass  # Keep values_conflict = True for non-numeric
                    if values_conflict:
                        conflicts.append(f"value {raw_name}={value} conflicts with {value_source}={merged['value']}")
            unit = item.get("unit")
            if unit:
                if merged["unit"] is None:
                    merged["unit"] = unit
                    unit_source = raw_name
                elif merged["unit"] != unit:
                    conflicts.append(f"unit {raw_name}={unit} conflicts with {unit_source}={merged['unit']}")
            method = item.get("method")
            if method:
                if merged["method"] is None:
                    merged["method"] = method
                    method_source = raw_name
                elif merged["method"] != method:
                    conflicts.append(f"method {raw_name}={method} conflicts with {method_source}={merged['method']}")
            rel_tol = item.get("rel_tol")
            if rel_tol is not None:
                merged["rel_tol"] = rel_tol if merged["rel_tol"] is None else min(merged["rel_tol"], rel_tol)
            abs_tol = item.get("abs_tol")
            if abs_tol is not None:
                merged["abs_tol"] = abs_tol if merged["abs_tol"] is None else min(merged["abs_tol"], abs_tol)
            if item.get("fixed"):
                merged["fixed"] = True
        if len(entries) > 1:
            if conflicts:
                warnings.warn(f"Conflicting aliases for '{name}': " + "; ".join(conflicts), UserWarning)
            else:
                warnings.warn(f"Merged aliases for '{name}': {[k for k, _ in entries]}", UserWarning)
        value = merged.get("value")
        unit = merged.get("unit")
        method = merged.get("method")
        rel_tol = merged.get("rel_tol")
        abs_tol = merged.get("abs_tol")
        fixed = bool(merged.get("fixed", False))
        input_source = "explicit" if value is not None else None
        if rel_tol is not None and abs_tol is not None:
            if abs_tol >= rel_tol:
                rel_tol = None
            else:
                abs_tol = None
        if unit is None:
            unit = default_units.get(name)
        if isinstance(value, str):
            try:
                value = float(value)
            except Exception:
                pass
        if value is not None and unit is not None:
            target = default_units.get(name)
            if target and unit != target:
                try:
                    import pint
                    ureg = pint.UnitRegistry()
                    qty = ureg.Quantity(value, unit).to(target)
                    value, unit = qty.magnitude, target
                except Exception:
                    pass
        var = parsed.get(name)
        if var is None:
            var = Variable(
                name=name,
                unit=unit,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                method=method,
                input_source=input_source,
                fixed=fixed,
            )
            parsed[name] = var
        if value is not None:
            var.values = [value]
            var.value_passes = [0]
            var.history = [{"pass_id": 0, "old": None, "new": value, "reason": "input"}]
            var.input_source = "explicit"
        if fixed:
            var.fixed = True
        if method and not var.method:
            var.method = method
        if rel_tol is not None:
            var.rel_tol = rel_tol
        if abs_tol is not None:
            var.abs_tol = abs_tol
        if unit and not var.unit:
            var.unit = unit
    return parsed


def validate_solver_tags(solver_tags: dict[str, Any] | None, *, log: logging.Logger | None = None) -> None:
    """Validate solver tags against solver_defaults.yaml keys."""
    if not solver_tags:
        return
    allowed = load_solver_defaults() or {}
    allowed_keys = set(allowed.keys())
    def _warn(msg: str) -> None:
        if log:
            log.warning(msg)
        else:
            warnings.warn(msg, UserWarning)
    for key in solver_tags:
        if key not in allowed_keys:
            _warn(f"Unknown solver tag '{key}'. Allowed: {sorted(allowed_keys)}")
    mode = solver_tags.get("mode")
    if mode is not None:
        if mode not in ("overwrite", "check"):
            _warn(f"Invalid solver mode '{mode}'. Allowed: ['overwrite', 'check']")
    solver = solver_tags.get("solver")
    if solver is not None:
        if solver not in ("lsq_compact",):
            _warn(f"Invalid solver '{solver}'. Allowed: ['lsq_compact']")
    lsq = solver_tags.get("lsq")
    if lsq is not None and not isinstance(lsq, dict):
        _warn("solver_tags.lsq must be a mapping")
    elif isinstance(lsq, dict):
        allowed_lsq = allowed.get("lsq", {})
        allowed_lsq_keys = set(allowed_lsq.keys()) if isinstance(allowed_lsq, dict) else set()
        for key in lsq:
            if key not in allowed_lsq_keys:
                _warn(f"Unknown lsq setting '{key}'. Allowed: {sorted(allowed_lsq_keys)}")


# Define what's exported when using "from fusdb.registry import *"
__all__ = [
    # Paths
    "REGISTRY_PATH",
    "TAGS_PATH",
    "VARIABLES_PATH",
    "SPECIES_PATH",
    "SOLVER_DEFAULTS_PATH",
    # Functions
    "load_yaml",
    "load_constants",
    "load_allowed_variables",
    "load_solver_defaults",
    "allowed_variable_constraints",
    "normalize_tag",
    "normalize_tags_to_tuple",
    "canonical_variable_name",
    "normalize_country",
    "parse_variables",
    "validate_solver_tags",
    # Constants from tags
    "ALLOWED_VARIABLES",
    "ALLOWED_SOLVING_ORDER",
    "ALLOWED_CONFINEMENT_MODES",
    "ALLOWED_REACTOR_FAMILIES",
    "ALLOWED_REACTOR_CONFIGURATIONS",
    "OPTIONAL_METADATA_FIELDS",
    "REQUIRED_FIELDS",
    "RESERVED_KEYS",
    # Physical constants (dynamically loaded from constants.yaml)
    "MEV_TO_J",
    "KEV_TO_J",
    "MU0",
    "DT_REACTION_ENERGY_J",
    "DT_ALPHA_ENERGY_J",
    "DT_N_ENERGY_J",
    "DD_T_ENERGY_J",
    "DD_HE3_ENERGY_J",
    "DD_P_ENERGY_J",
    "DD_N_ENERGY_J",
    "DHE3_ALPHA_ENERGY_J",
    "DHE3_P_ENERGY_J",
    "TT_REACTION_ENERGY_J",
]
