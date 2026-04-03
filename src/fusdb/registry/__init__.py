"""Registry module for allowed variables, tags, and constants."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import warnings
import numpy as np
import yaml
from ..utils import within_tolerance, normalize_tag, normalize_tags_to_tuple, normalize_country
from ..variable_class import Variable
from ..variable_util import make_variable

# Registry paths
REGISTRY_PATH = Path(__file__).resolve().parent
TAGS_PATH = REGISTRY_PATH / "allowed_tags.yaml"
VARIABLES_PATH = REGISTRY_PATH / "allowed_variables.yaml"
SPECIES_PATH = REGISTRY_PATH / "allowed_species.yaml"
SOLVER_DEFAULTS_PATH = REGISTRY_PATH / "solver_defaults.yaml"
ALLOWED_REACTIONS_PATH = REGISTRY_PATH / "allowed_reactions.yaml"

# Private caches
_ALLOWED_VARIABLES: dict[str, dict] | None = None
_ALIASES: dict[str, str] | None = None
_DEFAULT_UNITS: dict[str, str] | None = None
_ALLOWED_TAGS: dict[str, Any] | None = None
_ALLOWED_SPECIES: dict[str, dict[str, Any]] | None = None
_CONSTANTS: dict[str, float] | None = None
_SOLVER_DEFAULTS: dict[str, Any] | None = None
_ALLOWED_REACTIONS_DOCUMENT: dict[str, Any] | None = None
_ALLOWED_REACTIONS: dict[str, dict[str, Any]] | None = None


def load_constants() -> dict[str, float]:
    """Load physical constants from YAML. Args: none. Returns: dict."""
    global _CONSTANTS
    if _CONSTANTS is None:
        with (REGISTRY_PATH / "constants.yaml").open("r", encoding="utf-8") as handle:
            _CONSTANTS = yaml.safe_load(handle) or {}
    return _CONSTANTS


def load_allowed_species() -> dict[str, dict[str, Any]]:
    """Load allowed species metadata from YAML."""
    global _ALLOWED_SPECIES
    if _ALLOWED_SPECIES is None:
        with SPECIES_PATH.open("r", encoding="utf-8") as handle:
            _ALLOWED_SPECIES = yaml.safe_load(handle) or {}
    return _ALLOWED_SPECIES


def load_allowed_reactions_document() -> dict[str, Any]:
    """Load the raw reaction registry YAML document."""
    global _ALLOWED_REACTIONS_DOCUMENT
    if _ALLOWED_REACTIONS_DOCUMENT is None:
        with ALLOWED_REACTIONS_PATH.open("r", encoding="utf-8") as handle:
            _ALLOWED_REACTIONS_DOCUMENT = yaml.safe_load(handle) or {}
    return _ALLOWED_REACTIONS_DOCUMENT


def load_allowed_reactions() -> dict[str, dict[str, Any]]:
    """Load allowed reaction metadata from YAML."""
    global _ALLOWED_REACTIONS
    if _ALLOWED_REACTIONS is None:
        document = load_allowed_reactions_document()
        _ALLOWED_REACTIONS = {
            name: spec
            for name, spec in document.items()
            if isinstance(name, str) and name != "settings" and isinstance(spec, dict)
        }
    return _ALLOWED_REACTIONS


def load_reactivity_table_config() -> dict[str, Any]:
    """Load shared reactivity-table settings from the reaction registry."""
    document = load_allowed_reactions_document()
    settings = document.get("settings", {})
    return settings if isinstance(settings, dict) else {}


def load_solver_defaults() -> dict[str, Any]:
    """Load solver defaults. Args: none. Returns: dict."""
    global _SOLVER_DEFAULTS
    if _SOLVER_DEFAULTS is None:
        with SOLVER_DEFAULTS_PATH.open("r", encoding="utf-8") as handle:
            _SOLVER_DEFAULTS = yaml.safe_load(handle) or {}
    return _SOLVER_DEFAULTS


def load_allowed_variables() -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    """Load allowed variables. Args: none. Returns: (data, alias_map, default_units)."""
    global _ALLOWED_VARIABLES, _ALIASES, _DEFAULT_UNITS
    if _ALLOWED_VARIABLES is not None:
        return _ALLOWED_VARIABLES, _ALIASES or {}, _DEFAULT_UNITS or {}
    with (REGISTRY_PATH / "allowed_variables.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
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
    name = canonical_variable_name(name)
    data, _, _ = load_allowed_variables()
    return tuple((data.get(name, {}) or {}).get("constraints") or ())


def allowed_variable_soft_constraints(name: str) -> tuple[str, ...]:
    """Return soft constraints for an allowed variable. Args: name. Returns: tuple[str,...]."""
    name = canonical_variable_name(name)
    data, _, _ = load_allowed_variables()
    return tuple((data.get(name, {}) or {}).get("soft_constraints") or ())


def allowed_variable_ndim(name: str) -> int:
    """Return dimensionality for an allowed variable (0 if unspecified)."""
    name = canonical_variable_name(name)
    data, _, _ = load_allowed_variables()
    ndim = (data.get(name, {}) or {}).get("ndim", 0)
    try:
        return int(ndim)
    except Exception:
        return 0


def load_allowed_tags() -> dict[str, Any]:
    """Load allowed tags metadata. Args: none. Returns: dict."""
    global _ALLOWED_TAGS
    if _ALLOWED_TAGS is None:
        with (REGISTRY_PATH / "allowed_tags.yaml").open("r", encoding="utf-8") as handle:
            tags = yaml.safe_load(handle) or {}
        if "solving_order" in tags:
            tags["solving_order"] = {normalize_tag(k): v for k, v in (tags.get("solving_order") or {}).items()}
        for key in ("confinement_modes", "reactor_families", "reactor_configurations"):
            if key in tags:
                tags[key] = list(normalize_tags_to_tuple(tags.get(key) or []))
        _ALLOWED_TAGS = tags
    return _ALLOWED_TAGS

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
    """Parse raw YAML variables into canonical `Variable` objects.

    Args:
        variables: Raw mapping loaded from a reactor YAML file.

    Returns:
        Parsed variables keyed by canonical variable name.

    Raises:
        ValueError: If entries conflict or use unsupported value payloads.
    """
    _, _, default_units = load_allowed_variables()
    parsed: dict[str, Variable] = {}
    raw_vars = variables or {}

    # Expand the `fractions` block before the main parsing pass so the rest of
    # the function only has to deal with one normalized variable mapping.
    fractions = raw_vars.get("fractions") if isinstance(raw_vars, dict) else None
    if isinstance(fractions, dict):
        explicit_fraction_names = {
            canonical_variable_name(f"f_{species}")
            for species in fractions
        }
        explicit_density_names = {
            canonical_variable_name(str(raw_name))
            for raw_name in raw_vars
            if raw_name != "fractions"
            and canonical_variable_name(str(raw_name)).startswith("n_")
        }
        base: dict[str, Any] = {}
        for raw_name, entry in raw_vars.items():
            if raw_name == "fractions":
                continue
            canonical_name = canonical_variable_name(str(raw_name))
            if canonical_name in explicit_fraction_names:
                continue
            base[raw_name] = entry

        reference_name = None
        for candidate in ("n_i", "n_avg"):
            if candidate in base:
                reference_name = candidate
                break
        reference_entry = None if reference_name is None else base.get(reference_name)

        for species, frac in fractions.items():
            density_key = "n_imp" if str(species) == "Imp" else f"n_{species}"
            density_name = canonical_variable_name(density_key)
            if reference_entry is not None and density_name not in explicit_density_names:
                try:
                    scale = float(frac)
                except Exception as exc:
                    raise ValueError(
                        f"Fraction for species seed must be numeric: {frac!r}"
                    ) from exc
                if isinstance(reference_entry, dict):
                    raw_value = reference_entry.get("value")
                    if raw_value is None:
                        raise ValueError(
                            "fractions block requires a numeric 'n_i' or 'n_avg' value to seed species densities"
                        )
                    base[density_name] = {
                        "value": np.asarray(raw_value, dtype=float) * scale,
                        "unit": reference_entry.get("unit"),
                        "method": None,
                        "rel_tol": None,
                        "abs_tol": None,
                        "fixed": False,
                        "coord": reference_entry.get("coord"),
                    }
                else:
                    base[density_name] = np.asarray(reference_entry, dtype=float) * scale
                continue
            base[canonical_variable_name(f"f_{species}")] = {"value": frac}
        raw_vars = base

    # Group aliases first so the merge logic only has to reason about canonical names.
    grouped: dict[str, list[tuple[str, object]]] = {}
    if isinstance(raw_vars, dict):
        for raw_name, entry in raw_vars.items():
            if raw_name == "fractions":
                continue
            canonical_name = canonical_variable_name(str(raw_name))
            grouped.setdefault(canonical_name, []).append((str(raw_name), entry))

    # Merge all aliases for one canonical variable and then create/update the
    # corresponding `Variable` object once.
    for name, entries in grouped.items():
        merged: dict[str, object] = {
            "value": None,
            "unit": None,
            "method": None,
            "rel_tol": None,
            "abs_tol": None,
            "fixed": False,
            "coord": None,
        }
        value_source = unit_source = method_source = None
        coord_source = None
        conflicts: list[str] = []

        for raw_name, entry in entries:
            if isinstance(entry, dict):
                value = entry.get("value")
                if isinstance(value, dict):
                    raise ValueError(
                        "Profile dict payloads are no longer supported. "
                        "Use a numeric scalar or 1D numeric array for 'value'."
                    )
                unit = entry.get("unit")
                method = entry.get("method")
                rel_tol = entry.get("rel_tol")
                abs_tol = entry.get("abs_tol")
                fixed = bool(entry.get("fixed", False))
                coord = (
                    canonical_variable_name(str(entry.get("coord")))
                    if entry.get("coord")
                    else None
                )
            else:
                value = entry
                unit = None
                method = None
                rel_tol = None
                abs_tol = None
                fixed = False
                coord = None

            if value is not None:
                if merged["value"] is None:
                    merged["value"] = value
                    value_source = raw_name
                else:
                    values_same = merged["value"] is value
                    if not values_same:
                        values_same = within_tolerance(merged["value"], value)
                    if not values_same:
                        try:
                            left_arr = np.asarray(merged["value"], dtype=float)
                            right_arr = np.asarray(value, dtype=float)
                            values_same = bool(
                                left_arr.shape == right_arr.shape
                                and np.array_equal(left_arr, right_arr)
                            )
                        except Exception:
                            values_same = False
                    if not values_same:
                        try:
                            values_same = bool(merged["value"] == value)
                        except Exception:
                            values_same = False
                    if not values_same:
                        conflicts.append(
                            f"value {raw_name}={value} conflicts with {value_source}={merged['value']}"
                        )
            if unit:
                if merged["unit"] is None:
                    merged["unit"] = unit
                    unit_source = raw_name
                elif merged["unit"] != unit:
                    conflicts.append(f"unit {raw_name}={unit} conflicts with {unit_source}={merged['unit']}")
            if method:
                if merged["method"] is None:
                    merged["method"] = method
                    method_source = raw_name
                elif merged["method"] != method:
                    conflicts.append(f"method {raw_name}={method} conflicts with {method_source}={merged['method']}")
            if rel_tol is not None:
                merged["rel_tol"] = rel_tol if merged["rel_tol"] is None else min(merged["rel_tol"], rel_tol)
            if abs_tol is not None:
                merged["abs_tol"] = abs_tol if merged["abs_tol"] is None else min(merged["abs_tol"], abs_tol)
            if fixed:
                merged["fixed"] = True
            if coord:
                if merged["coord"] is None:
                    merged["coord"] = coord
                    coord_source = raw_name
                elif merged["coord"] != coord:
                    conflicts.append(
                        f"coord {raw_name}={coord} conflicts with {coord_source}={merged['coord']}"
                    )
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
        coord = merged.get("coord")
        input_source = "explicit" if value is not None else None
        ndim = allowed_variable_ndim(name)
        is_profile = ndim == 1 and value is not None
        if ndim == 1 and isinstance(value, list):
            try:
                value = np.asarray(value, dtype=float)
            except Exception:
                pass
        if ndim == 0 and isinstance(value, dict):
            raise ValueError(f"Variable '{name}' does not allow mapping/profile inputs.")
        if rel_tol is not None and abs_tol is not None:
            if abs_tol >= rel_tol:
                rel_tol = None
            else:
                abs_tol = None
        if unit is None:
            unit = default_units.get(name)
        if not is_profile and isinstance(value, str):
            try:
                value = float(value)
            except Exception:
                pass
        if not is_profile and value is not None and unit is not None:
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
            kwargs = dict(
                name=name,
                unit=unit,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                method=method,
                input_source=input_source,
                fixed=fixed,
                ndim=ndim,
            )
            if ndim == 1 and coord:
                kwargs["coord"] = coord
            var = make_variable(**kwargs)
            parsed[name] = var
        if value is not None:
            var.add_value(value, as_input=True)
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

    # Keep validation explicit so every branch emits the exact warning message
    # without routing through a local wrapper.
    for key in solver_tags:
        if key not in allowed_keys:
            msg = f"Unknown solver tag '{key}'. Allowed: {sorted(allowed_keys)}"
            if log:
                log.warning(msg)
            else:
                warnings.warn(msg, UserWarning)
    mode = solver_tags.get("mode")
    if mode is not None:
        if mode not in ("overwrite", "check"):
            msg = f"Invalid solver mode '{mode}'. Allowed: ['overwrite', 'check']"
            if log:
                log.warning(msg)
            else:
                warnings.warn(msg, UserWarning)
    solver = solver_tags.get("solver")
    if solver is not None:
        if solver not in ("lsq_compact",):
            msg = f"Invalid solver '{solver}'. Allowed: ['lsq_compact']"
            if log:
                log.warning(msg)
            else:
                warnings.warn(msg, UserWarning)
    lsq = solver_tags.get("lsq")
    if lsq is not None and not isinstance(lsq, dict):
        msg = "solver_tags.lsq must be a mapping"
        if log:
            log.warning(msg)
        else:
            warnings.warn(msg, UserWarning)
    elif isinstance(lsq, dict):
        allowed_lsq = allowed.get("lsq", {})
        allowed_lsq_keys = set(allowed_lsq.keys()) if isinstance(allowed_lsq, dict) else set()
        for key in lsq:
            if key not in allowed_lsq_keys:
                msg = f"Unknown lsq setting '{key}'. Allowed: {sorted(allowed_lsq_keys)}"
                if log:
                    log.warning(msg)
                else:
                    warnings.warn(msg, UserWarning)
