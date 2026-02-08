"""Default reactor inputs and fallback relations for missing YAML values.
Variables dict: {name: Variable}.
Variable: dataclass with fields name, values, value_passes, history, unit, rel_tol, abs_tol,
method, input_source, fixed.
"""

from __future__ import annotations

from fusdb.relation_class import Relation
from fusdb.relation_util import build_sympy_expr, function_inputs
from fusdb.utils import normalize_tags_to_tuple
from fusdb.variable_class import Variable


_FRACTION_DEFAULTS: dict[str, float] = {
    "f_D": 0.5,
    "f_T": 0.5,
    "f_He3": 0.0,
    "f_He4": 0.0,
}
_FRACTION_KEYS = tuple(_FRACTION_DEFAULTS.keys())

def _build_relation(name: str, output: str, func, *, tags: tuple[str, ...] = ("plasma",)) -> Relation:
    """Build a Relation without registering it globally."""
    inputs = function_inputs(func)
    expr, symbols = build_sympy_expr(func, inputs, output)
    return Relation(
        name=name,
        output=output,
        func=func,
        inputs=inputs,
        tags=normalize_tags_to_tuple(tags),
        sympy_expr=expr,
        _sympy_symbols=symbols,
    )


def _set_default_input(variables: dict[str, "Variable"], name: str, value: float) -> None:
    """Set a default input value when missing. Args: variables, name, value. Returns: None."""
    var = variables.get(name)
    if var is None:
        var = Variable(
            name=name,
            method="default",
            input_source="default",
        )
        variables[name] = var
    if Variable.get_from_dict(variables, name, mode="input") is None:
        var.values = [value]
        var.value_passes = [0]
        var.history = [{"pass_id": 0, "old": None, "new": value, "reason": "default"}]
        if var.method is None:
            var.method = "default"
        var.input_source = "default"


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
    default_relations: list[Relation] = []
    seen_keys = {(rel.name, rel.output) for rel in (relations or ())}

    def _add_default_relation(name: str, output: str, func, *, tags: tuple[str, ...] = ("plasma",)) -> None:
        key = (name, output)
        if key in seen_keys:
            return
        default_relations.append(_build_relation(name, output, func, tags=tags))
        seen_keys.add(key)

    # Gather input values for variables that already exist in the dictionary.
    input_values = {name: Variable.get_from_dict(variables, name, mode="input") for name in variables}
    
    def has_input(name: str) -> bool:
        check = input_values.get(name) is not None
        return check
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
            "Default: n_avg = n_e",
            "n_avg",
            lambda n_e: n_e,
        )
    if not has_input("n_e") and not has_input("n_i") and has_input("n_avg"):
        _add_default_relation(
            "Default: n_e = n_avg",
            "n_e",
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
                lambda T_i: T_i,
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
                lambda T_e: T_e,
            )

    # Fill missing T_avg
    if not has_Tavg:
        if has_Te:
            _add_default_relation(
                "Default: T_avg = T_e",
                "T_avg",
                lambda T_e: T_e,
            )
        elif has_Ti:
            _add_default_relation(
                "Default: T_avg = T_i",
                "T_avg",
                lambda T_i: T_i,
            )

    ####################### PLASMA COMPOSITION DEFAULTS #######################################
    
    # Default species particle confinement times to generic tau_p if provided.
    if has_input("tau_p"):
        tau_p = input_values.get("tau_p")
        for species in ("D", "T", "He3", "He4"):
            name = f"tau_p_{species}"
            if Variable.get_from_dict(variables, name, mode="input") is None:
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
