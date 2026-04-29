"""Default reactor inputs and fallback relations for missing YAML values.
Variables dict: {name: Variable}.
Variable: dataclass with compact `input_value` and `current_value` state.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable

from ..relation_class import Relation
from ..utils import mean_profile, normalize_tags_to_tuple, scalarize_value, within_tolerance
from ..variable_class import Variable
from . import allowed_variable_ndim, canonical_variable_name, load_allowed_reactions


_SPECIES_DENSITY_DEFAULTS: dict[str, float] = {
    "n_D": 0.5,
    "n_T": 0.5,
    "n_He3": 0.0,
    "n_He4": 0.0,
    "n_imp": 0.0,
}
_SPECIES_DENSITY_KEYS = tuple(_SPECIES_DENSITY_DEFAULTS.keys())
_SPECIES_TAU_SUFFIXES = ("D", "T", "He3", "He4", "Imp")


def _build_relation(name: str, output: str, func, *, tags: tuple[str, ...] = ("plasma",)) -> Relation:
    """Build a Relation without registering it globally.

    Args:
        name: Relation display name.
        output: Output variable name.
        func: Callable implementing the fallback relation.
        tags: Optional grouping tags.

    Returns:
        One non-registered Relation object.
    """
    # Keep only required named parameters as relation inputs.
    # This excludes captured/default helper args (for example `_fraction=...`).
    inputs: list[str] = []
    for param_name, param in inspect.signature(func).parameters.items():
        if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if param.default is not inspect._empty:
            continue
        inputs.append(param_name)

    tags_with_assumption = normalize_tags_to_tuple((*tags, "assumption"))
    return Relation.from_callable(
        name=name,
        func=func,
        target=output,
        inputs=tuple(inputs),
        tags=tags_with_assumption,
    )


def _set_default_input(variables: dict[str, "Variable"], name: str, value: object) -> None:
    """Set a default input value when missing. Args: variables, name, value. Returns: None."""
    var = variables.get(name)
    if var is None:
        var = Variable.make(
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
        var = Variable.make(
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
    seen_keys = {
        (rel.name, output)
        for rel in (relations or ())
        for output in rel.outputs
    }

    def _ensure_variable_exists(name: str) -> None:
        """Ensure a variable object exists for relation-only defaults."""
        if name in variables:
            return
        variables[name] = Variable.make(
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
    def has_input(name: str) -> bool:
        var = variables.get(name)
        return bool(var is not None and var.input_value is not None)

    def _profile_or_value(value: object) -> object:
        return scalarize_value(value)

    def _add_first_matching_default(
        output: str,
        rules: tuple[tuple[Callable[[], bool], str, Callable[..., object]], ...],
    ) -> None:
        """Add the first matching default relation for one output variable."""
        if has_input(output):
            return
        for condition, rel_name, rel_func in rules:
            if condition():
                _add_default_relation(rel_name, output, rel_func)
                return

    def _seed_species_from_reference(reference_name: str) -> bool:
        """Seed species densities from one explicit total density reference.

        Returns:
            ``True`` when the reference exists and this branch is considered handled.
        """
        if not has_input(reference_name):
            return False
        reference_value = scalarize_value(variables[reference_name].input_value)
        if reference_value is not None:
            for density_name, fraction in _SPECIES_DENSITY_DEFAULTS.items():
                var = variables.get(density_name)
                if var is not None and var.input_value is not None:
                    continue
                _set_default_input(variables, density_name, fraction * float(reference_value))
        return True

    ####################### GEOMETRY DEFAULTS ###########################################
    # Ensure squareness exists to avoid expensive inversion when missing.
    if not has_input("squareness"):
        _set_default_input(variables, "squareness", 0.0)
                
    ####################### DENSITY DEFAULTS ###########################################

    # Register simple density fallback relations from one compact rule table.
    density_fallback_rules = (
        ("Default: n_la = n_avg", "n_la", lambda n_avg: n_avg),
        ("Default: n_avg = mean(n_e)", "n_avg", lambda n_e: _profile_or_value(n_e)),
        ("Default: n_e = n_avg", "n_e", lambda n_avg: n_avg),
        ("Default: n_i = n_avg", "n_i", lambda n_avg: n_avg),
    )
    for rel_name, output_name, rel_func in density_fallback_rules:
        if has_input(output_name):
            continue
        _add_default_relation(rel_name, output_name, rel_func)
        
    ####################### TEMPERATURE DEFAULTS #######################################
    # Fill T_e, T_i, and T_avg from one explicit first-match policy each.
    _add_first_matching_default(
        "T_e",
        (
            (lambda: has_input("T_avg"), "Default: T_e = T_avg", lambda T_avg: T_avg),
            (lambda: has_input("T_i"), "Default: T_e = T_i", lambda T_i: _profile_or_value(T_i)),
            (lambda: True, "Default: T_e = T_avg", lambda T_avg: T_avg),
        ),
    )
    _add_first_matching_default(
        "T_i",
        (
            (lambda: has_input("T_avg"), "Default: T_i = T_avg", lambda T_avg: T_avg),
            (lambda: has_input("T_e"), "Default: T_i = T_e", lambda T_e: _profile_or_value(T_e)),
            (lambda: True, "Default: T_i = T_avg", lambda T_avg: T_avg),
        ),
    )
    _add_first_matching_default(
        "T_avg",
        (
            (lambda: has_input("T_e"), "Default: T_avg = mean(T_e)", lambda T_e: _profile_or_value(T_e)),
            (lambda: has_input("T_i"), "Default: T_avg = mean(T_i)", lambda T_i: _profile_or_value(T_i)),
        ),
    )

    # Consistency checks between profiles and averages (non-fatal).
    def _warn_profile_mismatch(profile_name: str, avg_name: str) -> None:
        profile_var = variables.get(profile_name)
        avg_var = variables.get(avg_name)
        profile_val = None if profile_var is None else profile_var.input_value
        avg_val = None if avg_var is None else avg_var.input_value
        if profile_val is None or avg_val is None:
            return
        mean_val = mean_profile(profile_val)
        if mean_val is None:
            return
        try:
            avg_scalar = float(avg_val)
        except Exception:
            return
        if not within_tolerance(mean_val, avg_scalar, rel_tol=1e-2):
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

    for reaction, spec in load_allowed_reactions().items():
        sigmav_variable = spec.get("sigmav_variable")
        default_method = spec.get("default_method")
        if isinstance(sigmav_variable, str) and isinstance(default_method, str):
            _set_default_method(variables, sigmav_variable, f"{reaction} reactivity {default_method}")

    ####################### PLASMA COMPOSITION DEFAULTS #######################################
    
    # Default species particle confinement times to generic tau_p if provided.
    if has_input("tau_p"):
        tau_p = variables["tau_p"].input_value
        for species in _SPECIES_TAU_SUFFIXES:
            name = f"tau_p_{species}"
            var = variables.get(name)
            if var is None or var.input_value is None:
                _set_default_input(variables, name, float(tau_p))

    density_inputs = {key: variables.get(key) for key in _SPECIES_DENSITY_KEYS}
    has_explicit_density = any(
        var is not None and var.input_value is not None and var.input_source == "explicit"
        for var in density_inputs.values()
    )

    # Step 1: if the user provided any explicit species density, keep missing species at zero.
    if has_explicit_density:
        for name in _SPECIES_DENSITY_KEYS:
            var = variables.get(name)
            if var is not None and var.input_value is not None:
                continue
            _set_default_input(variables, name, 0.0)
        return default_relations

    # Step 2: when total density is provided as input, seed species directly from that input.
    if _seed_species_from_reference("n_i"):
        return default_relations

    if _seed_species_from_reference("n_avg"):
        return default_relations

    # Step 3: when total density is not yet known, add fallback assumptions tied to n_i.
    fuel_species_relations = (
        ("Default: n_D = 0.5 * n_i", "n_D", lambda n_i: 0.5 * n_i),
        ("Default: n_T = 0.5 * n_i", "n_T", lambda n_i: 0.5 * n_i),
    )
    for rel_name, output_name, rel_func in fuel_species_relations:
        if has_input(output_name):
            continue
        _add_default_relation(rel_name, output_name, rel_func)

    # Keep non-fuel tracked species off unless user provides them explicitly.
    for name in ("n_He3", "n_He4", "n_imp"):
        if not has_input(name):
            _set_default_input(variables, name, 0.0)
    return default_relations
