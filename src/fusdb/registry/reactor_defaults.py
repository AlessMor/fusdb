"""Default reactor inputs and fallback relations for missing YAML values.
Variables dict: {name: Variable}.
Variable: dataclass with compact `input_value` and `current_value` state.
"""

from __future__ import annotations

import inspect
from fusdb.relation_class import Relation
import warnings
import numpy as np

from fusdb.utils import as_profile_array, normalize_tags_to_tuple, within_tolerance
from fusdb.variable_class import Variable
from fusdb.variable_util import make_variable
from fusdb.registry import allowed_variable_ndim, canonical_variable_name


_FRACTION_DEFAULTS: dict[str, float] = {
    "f_D": 0.5,
    "f_T": 0.5,
    "f_He3": 0.0,
    "f_He4": 0.0,
}
_FRACTION_KEYS = tuple(_FRACTION_DEFAULTS.keys())
_REACTIVITY_DEFAULT_METHODS: dict[str, str] = {
    "sigmav_DT": "DT reactivity BoschHale",
    "sigmav_DDn": "DDn reactivity BoschHale",
    "sigmav_DDp": "DDp reactivity BoschHale",
    "sigmav_DHe3": "DHe3 reactivity BoschHale",
    "sigmav_TT": "TT reactivity CF88",
    "sigmav_He3He3": "He3He3 reactivity CF88",
    "sigmav_THe3": "THe3 reactivity CF88",
}

def _build_relation(name: str, output: str, func, *, tags: tuple[str, ...] = ("plasma",)) -> Relation:
    """Build a Relation without registering it globally."""
    # NOTE: This intentionally keeps all signature parameter names, including *args/**kwargs names;
    # possibly add a filter for allowed variable names in the future. (but be careful not to reject part of the relation signature!)
    inputs = tuple(inspect.signature(func).parameters)
    return Relation.from_callable(
        name=name,
        func=func,
        target=output,
        inputs=inputs,
        tags=normalize_tags_to_tuple(tags),
    )


def _set_default_input(variables: dict[str, "Variable"], name: str, value: object) -> None:
    """Set a default input value when missing. Args: variables, name, value. Returns: None."""
    var = variables.get(name)
    if var is None:
        var = make_variable(
            name=name,
            method="default",
            input_source="default",
            ndim=allowed_variable_ndim(name),
        )
        variables[name] = var
    if var.input_value is None:
        var.add_value(value, as_input=True)
        if var.method is None:
            var.method = "default"
        var.input_source = "default"


def _set_default_method(variables: dict[str, "Variable"], name: str, method: str) -> None:
    """Ensure a variable exists and has a default method when none was specified."""
    var = variables.get(name)
    if var is None:
        var = make_variable(
            name=name,
            method=method,
            ndim=allowed_variable_ndim(name),
        )
        variables[name] = var
        return
    if var.method is None:
        var.method = method


def apply_reactor_defaults(
    variables: dict[str, "Variable"],
    relations: list[Relation] | None = None,
) -> list[Relation]:
    """Apply default inputs and fallback relations in-place.

    Args:
        variables: variables dict with per-variable entries.
        relations: existing relations to avoid duplicates.

    Returns:
        List of default Relation objects (not registered globally).
    """
    normalized_variables: dict[str, Variable] = {}
    for raw_name, var in list(variables.items()):
        canonical_name = canonical_variable_name(getattr(var, "name", raw_name))
        if canonical_name not in normalized_variables or raw_name == canonical_name:
            normalized_variables[canonical_name] = var
    if normalized_variables.keys() != variables.keys():
        variables.clear()
        variables.update(normalized_variables)

    default_relations: list[Relation] = []
    seen_keys = {(rel.name, (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None))) for rel in (relations or ())}

    def _ensure_variable_exists(name: str) -> None:
        """Ensure a variable object exists for relation-only defaults."""
        if name in variables:
            return
        variables[name] = make_variable(
            name=name,
            method="default",
            ndim=allowed_variable_ndim(name),
        )

    def _add_default_relation(name: str, output: str, func, *, tags: tuple[str, ...] = ("plasma",)) -> None:
        key = (name, output)
        if key in seen_keys:
            return
        _ensure_variable_exists(output)
        default_relations.append(_build_relation(name, output, func, tags=tags))
        seen_keys.add(key)

    # Gather input values for variables that already exist in the dictionary.
    input_values = {name: var.input_value for name, var in variables.items()}
    
    def has_input(name: str) -> bool:
        check = input_values.get(name) is not None
        return check

    def _profile_or_value(value: object) -> object:
        arr = as_profile_array(value)
        mean_val = float(np.mean(arr)) if arr is not None else None
        return mean_val if mean_val is not None else value
    ####################### GEOMETRY DEFAULTS ###########################################
    # Ensure squareness exists to avoid expensive inversion when missing.
    if not has_input("squareness"):
        _set_default_input(variables, "squareness", 0.0)
                
    ####################### DENSITY DEFAULTS ###########################################

    # Register fallback relations only when their outputs are missing.
    if not has_input("n_la"):
        _add_default_relation(
            "Default: n_la = n_avg",
            "n_la",
            lambda n_avg: n_avg,
        )
    if not has_input("n_avg"):
        _add_default_relation(
            "Default: n_avg = mean(n_e)",
            "n_avg",
            lambda n_e: _profile_or_value(n_e),
        )
    if not has_input("n_e"):
        _add_default_relation(
            "Default: n_e = n_avg",
            "n_e",
            lambda n_avg: n_avg,
        )
    if not has_input("n_i"):
        _add_default_relation(
            "Default: n_i = n_avg",
            "n_i",
            lambda n_avg: n_avg,
        )
        
    ####################### TEMPERATURE DEFAULTS #######################################
    has_Te = has_input("T_e")
    has_Ti = has_input("T_i")
    has_Tavg = has_input("T_avg")

    # Fill missing T_e / T_i
    if not has_Te:
        if has_Tavg:
            _add_default_relation(
                "Default: T_e = T_avg",
                "T_e",
                lambda T_avg: T_avg,
            )
        elif has_Ti:
            _add_default_relation(
                "Default: T_e = T_i",
                "T_e",
                lambda T_i: _profile_or_value(T_i),
            )
        else:
            _add_default_relation(
                "Default: T_e = T_avg",
                "T_e",
                lambda T_avg: T_avg,
            )
    if not has_Ti:
        if has_Tavg:
            _add_default_relation(
                "Default: T_i = T_avg",
                "T_i",
                lambda T_avg: T_avg,
            )
        elif has_Te:
            _add_default_relation(
                "Default: T_i = T_e",
                "T_i",
                lambda T_e: _profile_or_value(T_e),
            )
        else:
            _add_default_relation(
                "Default: T_i = T_avg",
                "T_i",
                lambda T_avg: T_avg,
            )

    # Fill missing T_avg
    if not has_Tavg:
        if has_Te:
            _add_default_relation(
                "Default: T_avg = mean(T_e)",
                "T_avg",
                lambda T_e: _profile_or_value(T_e),
            )
        elif has_Ti:
            _add_default_relation(
                "Default: T_avg = mean(T_i)",
                "T_avg",
                lambda T_i: _profile_or_value(T_i),
            )

    # Consistency checks between profiles and averages (non-fatal).
    def _warn_profile_mismatch(profile_name: str, avg_name: str) -> None:
        profile_var = variables.get(profile_name)
        avg_var = variables.get(avg_name)
        profile_val = None if profile_var is None else profile_var.input_value
        avg_val = None if avg_var is None else avg_var.input_value
        if profile_val is None or avg_val is None:
            return
        arr = as_profile_array(profile_val)
        mean_val = float(np.mean(arr)) if arr is not None else None
        if mean_val is None:
            return
        try:
            avg_scalar = float(avg_val)
        except Exception:
            return
        if not within_tolerance(mean_val, avg_scalar, rel_tol=1e-2, abs_tol=1e-8):
            warnings.warn(
                f"Profile/average mismatch for {avg_name}: mean({profile_name})={mean_val:.6g}, {avg_name}={avg_scalar:.6g}",
                UserWarning,
            )

    _warn_profile_mismatch("n_e", "n_avg")
    _warn_profile_mismatch("T_e", "T_avg")
    _warn_profile_mismatch("T_i", "T_avg")

    ####################### POWER DEFAULTS #######################################

    # Default ohmic heating to zero if not specified.
    if not has_input("P_ohmic"):
        _set_default_input(variables, "P_ohmic", 0.0)

    ####################### REACTIVITY METHOD DEFAULTS #######################################

    for name, method in _REACTIVITY_DEFAULT_METHODS.items():
        _set_default_method(variables, name, method)

    ####################### PLASMA COMPOSITION DEFAULTS #######################################
    
    # Default species particle confinement times to generic tau_p if provided.
    if has_input("tau_p"):
        tau_p = input_values.get("tau_p")
        for species in ("D", "T", "He3", "He4"):
            name = f"tau_p_{species}"
            var = variables.get(name)
            if var is None or var.input_value is None:
                _set_default_input(variables, name, float(tau_p))
                
    # If no fractions are explicit, set D/T defaults (and others to 0).
    fraction_inputs = {key: input_values.get(key) for key in _FRACTION_KEYS}
    has_explicit_fraction = False
    for key, value in fraction_inputs.items():
        var = variables.get(key)
        if var is not None and value is not None and var.input_source == "explicit":
            has_explicit_fraction = True
            break
    
    if not has_explicit_fraction:
        for name, value in _FRACTION_DEFAULTS.items():
            _set_default_input(variables, name, value)
        return default_relations

    # If any fractions are explicit, default missing ones to 0.0.
    for name in _FRACTION_KEYS:
        var = variables.get(name)
        if var is None or fraction_inputs.get(name) is None:
            _set_default_input(variables, name, 0.0)
    return default_relations
