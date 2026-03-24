"""Utilities shared by :class:`fusdb.relation_class.Relation` objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping
import inspect
import types
import warnings

import sympy as sp

from .utils import normalize_tags_to_tuple

if TYPE_CHECKING:
    from .relation_class import Relation

_RELATION_REGISTRY: list["Relation"] = []
def try_sympify_expression(
    expression: str,
    *,
    local_symbols: Mapping[str, object] | None = None,
    context: str | None = None,
    strict: bool = False,
) -> sp.Expr | None:
    """Parse an expression with sympy, warning/raising on failure."""
    try:
        return sp.sympify(expression, locals=dict(local_symbols or {}))
    except Exception as exc:
        where = f" for {context}" if context else ""
        msg = (
            f"Sympy could not parse expression{where}: {expression!r} "
            f"({type(exc).__name__}: {exc})"
        )
        if strict:
            raise ValueError(msg) from exc
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return None


def build_symbolic_model(
    func: Callable,
    arg_names: Iterable[str],
    preferred_target: str | None,
    *,
    relation_name: str | None = None,
    strict: bool = False,
) -> tuple[sp.Expr | None, dict[str, sp.Symbol] | None]:
    """Build implicit equation ``preferred_target - f(arg_names)``."""
    if preferred_target is None:
        return None, None

    arg_tuple = tuple(arg_names)
    symbols = {name: sp.Symbol(name, real=True) for name in (*arg_tuple, preferred_target)}

    direct_exc: Exception | None = None
    try:
        expr = func(*[symbols[name] for name in arg_tuple])
        return symbols[preferred_target] - expr, symbols
    except Exception as exc:
        direct_exc = exc

    # Inline globals patching for module-style and direct math/numpy names.
    patched_globals = dict(func.__globals__)
    patched_globals.setdefault("__builtins__", func.__globals__.get("__builtins__", __builtins__))
    sympy_names: dict[str, object] = {
        "pi": sp.pi,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "log10": sp.log,
        "log2": sp.log,
        "power": sp.Pow,
        "abs": sp.Abs,
        "fabs": sp.Abs,
        "floor": sp.floor,
        "ceil": getattr(sp, "ceil", sp.ceiling),
    }
    module_proxy = types.SimpleNamespace(
        **{
            **{name: getattr(sp, name) for name in dir(sp)},
            **sympy_names,
        }
    )
    for module_name in ("math", "np", "numpy"):
        if module_name in patched_globals:
            patched_globals[module_name] = module_proxy
    for name, sympy_obj in sympy_names.items():
        if name in patched_globals:
            patched_globals[name] = sympy_obj

    proxy_exc: Exception | None = None
    try:
        patched_func = types.FunctionType(
            func.__code__,
            patched_globals,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        # Preserve keyword-only defaults for callables that use ``def f(*args, key=...)``.
        patched_func.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
        expr = patched_func(*[symbols[name] for name in arg_tuple])
        return symbols[preferred_target] - expr, symbols
    except Exception as exc:
        proxy_exc = exc

    rel_label = relation_name or getattr(func, "__name__", "<unknown>")
    direct_label = (
        "n/a"
        if direct_exc is None
        else f"{type(direct_exc).__name__}: {direct_exc}"
    )
    proxy_label = (
        "n/a"
        if proxy_exc is None
        else f"{type(proxy_exc).__name__}: {proxy_exc}"
    )
    msg = (
        f"Could not convert relation '{rel_label}' to sympy expression for target "
        f"'{preferred_target}'. direct={direct_label}; proxy={proxy_label}"
    )
    if strict:
        raise ValueError(msg) from (proxy_exc or direct_exc)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return None, symbols


def normalize_relation_definition(relation: "Relation") -> None:
    """Canonicalize and validate a relation instance in place."""
    from .registry import canonical_variable_name

    canonical_inputs = tuple(canonical_variable_name(str(name)) for name in (relation.inputs or ()))
    canonical_outputs = tuple(canonical_variable_name(str(name)) for name in (relation.outputs or ()))
    if not canonical_outputs:
        raise ValueError(f"Relation '{relation.name}' must define at least one output")
    if not callable(relation.forward):
        raise ValueError(f"Relation '{relation.name}' forward callable is not callable")

    canonical_symbols: dict[str, sp.Symbol] = {}
    for raw_name, symbol in (relation.symbols or {}).items():
        c_name = canonical_variable_name(str(raw_name))
        if c_name in canonical_symbols:
            continue
        if isinstance(symbol, sp.Symbol) and symbol.name == c_name:
            canonical_symbols[c_name] = symbol
        else:
            canonical_symbols[c_name] = sp.Symbol(c_name, real=True)
    for name in (*canonical_inputs, *canonical_outputs):
        canonical_symbols.setdefault(name, sp.Symbol(name, real=True))

    canonical_solvers: dict[str, tuple[tuple[str, ...], Callable]] = {}
    for raw_target, spec in (relation.solvers or {}).items():
        try:
            arg_names, fn = spec
        except Exception as exc:
            raise ValueError(
                f"Relation '{relation.name}' has invalid solver entry for "
                f"target '{raw_target}': expected (inputs, callable)"
            ) from exc
        c_target = canonical_variable_name(str(raw_target))
        c_args = tuple(canonical_variable_name(str(name)) for name in arg_names)
        if not callable(fn):
            raise ValueError(
                f"Relation '{relation.name}' solver for target '{c_target}' is not callable"
            )
        canonical_solvers[c_target] = (c_args, fn)
        canonical_symbols.setdefault(c_target, sp.Symbol(c_target, real=True))
        for name in c_args:
            canonical_symbols.setdefault(name, sp.Symbol(name, real=True))

    canonical_overrides = {
        canonical_variable_name(str(name)): fn
        for name, fn in (relation.inverse_functions or {}).items()
    }
    if len(canonical_outputs) > 1:
        if canonical_solvers:
            raise ValueError(
                f"Relation '{relation.name}' with multiple outputs must be forward-only"
            )
        if canonical_overrides:
            raise ValueError(
                f"Relation '{relation.name}' with multiple outputs cannot define inverse functions"
            )
        if relation.sympy_expression is not None:
            raise ValueError(
                f"Relation '{relation.name}' with multiple outputs cannot define a symbolic expression"
            )

    object.__setattr__(relation, "inputs", canonical_inputs)
    object.__setattr__(relation, "outputs", canonical_outputs)
    object.__setattr__(relation, "tags", tuple(relation.tags))
    object.__setattr__(relation, "constraints", tuple(relation.constraints))
    object.__setattr__(relation, "soft_constraints", tuple(relation.soft_constraints))
    object.__setattr__(relation, "initial_guesses", dict(relation.initial_guesses or {}))
    object.__setattr__(relation, "inverse_functions", canonical_overrides)
    object.__setattr__(relation, "symbols", canonical_symbols)
    object.__setattr__(relation, "solvers", canonical_solvers)


def relation_input_names(relation: "Relation", output: str | None = None) -> tuple[str, ...]:
    """Return ordered input names for one numeric target."""
    target = relation.outputs[0] if output is None else output
    if target in relation.outputs:
        return relation.inputs
    solver = relation.solvers.get(target)
    if solver is not None:
        return solver[0]
    return tuple(name for name in relation.symbols if name != target)


def build_relation_from_callable(
    *,
    name: str,
    func: Callable,
    target: str | None = None,
    outputs: Iterable[str] | None = None,
    symbols: Iterable[str] | Mapping[str, object] | None = None,
    inputs: Iterable[str] | None = None,
    tags: Iterable[str] = (),
    rel_tol_default: float | None = None,
    abs_tol_default: float | None = None,
    constraints: Iterable[str] = (),
    soft_constraints: Iterable[str] = (),
    initial_guesses: dict[str, Callable] | None = None,
    inverse_functions: dict[str, Callable] | None = None,
    solvers: dict[str, tuple[tuple[str, ...], Callable]] | None = None,
    strict_symbolic: bool = False,
) -> "Relation":
    """Create a relation from a python callable and derive symbolic metadata."""
    from .registry import canonical_variable_name
    from .relation_class import Relation

    relation_name = name
    if inputs is None:
        input_tuple = tuple(inspect.signature(func).parameters)
    else:
        input_tuple = tuple(inputs)
    input_tuple = tuple(canonical_variable_name(str(name)) for name in input_tuple)

    if outputs is None:
        output_tuple = () if target is None else (target,)
    else:
        output_tuple = tuple(outputs)
    output_tuple = tuple(canonical_variable_name(str(name)) for name in output_tuple)

    if len(output_tuple) > 1 and target is not None:
        raise ValueError(
            f"Relation '{relation_name}' cannot define both target={target!r} "
            f"and multiple outputs={output_tuple!r}"
        )
    if not output_tuple:
        raise ValueError(f"Relation '{relation_name}' must define at least one output")

    preferred_target = output_tuple[0]
    multi_output = len(output_tuple) > 1

    base_vars: list[str] = []
    provided_symbols: dict[str, object] = {}
    if isinstance(symbols, Mapping):
        provided_symbols = dict(symbols)
        base_vars.extend(str(name) for name in symbols)
    elif symbols is not None:
        base_vars.extend(str(name) for name in symbols)
    for input_name in input_tuple:
        if input_name not in base_vars:
            base_vars.append(input_name)
    for output_name in output_tuple:
        if output_name not in base_vars:
            base_vars.append(output_name)

    if multi_output:
        expr = None
        symbols = {name: sp.Symbol(name, real=True) for name in base_vars}
    else:
        expr, symbols = build_symbolic_model(
            func,
            input_tuple,
            preferred_target,
            relation_name=relation_name,
            strict=strict_symbolic,
        )
    symbols_map = symbols if symbols is not None else {name: sp.Symbol(name, real=True) for name in base_vars}
    for raw_name, symbol in provided_symbols.items():
        symbols_map.setdefault(str(raw_name), symbol)
    for var_name in base_vars:
        symbols_map.setdefault(var_name, sp.Symbol(var_name, real=True))

    direct_solvers = dict(solvers or {})
    if multi_output and direct_solvers:
        raise ValueError(
            f"Relation '{relation_name}' with multiple outputs must be forward-only"
        )
    if multi_output and inverse_functions:
        raise ValueError(
            f"Relation '{relation_name}' with multiple outputs cannot define inverse functions"
        )

    constraints_tuple = tuple(constraints)
    soft_constraints_tuple = tuple(soft_constraints)
    for expr_str in constraints_tuple:
        try_sympify_expression(
            str(expr_str),
            local_symbols=symbols_map,
            context=f"relation '{relation_name}' hard constraints",
            strict=strict_symbolic,
        )
    for expr_str in soft_constraints_tuple:
        try_sympify_expression(
            str(expr_str),
            local_symbols=symbols_map,
            context=f"relation '{relation_name}' soft constraints",
            strict=strict_symbolic,
        )

    return Relation(
        name=relation_name,
        inputs=input_tuple,
        outputs=output_tuple,
        forward=func,
        tags=tuple(tags),
        rel_tol_default=rel_tol_default,
        abs_tol_default=abs_tol_default,
        constraints=constraints_tuple,
        soft_constraints=soft_constraints_tuple,
        initial_guesses=initial_guesses or {},
        inverse_functions=inverse_functions or {},
        sympy_expression=expr,
        symbols=symbols_map,
        solvers=direct_solvers,
    )


def _call_relation_solver(relation: "Relation", target: str, values: Mapping[str, object]) -> object:
    """Evaluate one direct solver mapping."""
    solver = relation.solvers.get(target)
    if solver is None:
        raise KeyError(f"Relation '{relation.name}' has no solver for target '{target}'")
    ordered, fn = solver
    return fn(*(values[name] for name in ordered))


def _call_relation_forward(relation: "Relation", values: Mapping[str, object]) -> object:
    """Evaluate the forward callable on canonical ordered inputs."""
    return relation.forward(*(values[name] for name in relation.inputs))


def apply_relation(relation: "Relation", values: Mapping[str, object]) -> dict[str, object]:
    """Evaluate the forward mapping and return output assignments."""
    result = _call_relation_forward(relation, values)
    if relation.is_multi_output:
        if not isinstance(result, Mapping):
            raise TypeError(
                f"Relation '{relation.name}' with multiple outputs must return a mapping"
            )
        from .registry import canonical_variable_name

        canonical_result = {
            canonical_variable_name(str(name)): value
            for name, value in result.items()
        }
        missing = [name for name in relation.outputs if name not in canonical_result]
        if missing:
            raise KeyError(
                f"Relation '{relation.name}' did not return outputs {missing}"
            )
        return {name: canonical_result[name] for name in relation.outputs}

    target = relation.outputs[0]
    if isinstance(result, Mapping):
        from .registry import canonical_variable_name

        canonical_result = {
            canonical_variable_name(str(name)): value
            for name, value in result.items()
        }
        if target not in canonical_result:
            raise KeyError(
                f"Relation '{relation.name}' did not return output '{target}'"
            )
        return {target: canonical_result[target]}
    return {target: result}


def evaluate_relation(
    relation: "Relation", values: Mapping[str, object], target: str | None = None
) -> object:
    """Evaluate a numeric function for the requested target."""
    eval_target = relation.outputs[0] if target is None else target
    if relation.is_multi_output:
        if target is None:
            return apply_relation(relation, values)
        return apply_relation(relation, values)[eval_target]
    if eval_target in relation.solvers:
        return _call_relation_solver(relation, eval_target, values)
    if eval_target not in relation.outputs:
        raise KeyError(f"Relation '{relation.name}' has no numeric function for target '{eval_target}'")
    return apply_relation(relation, values)[eval_target]


def inverse_solver_for_relation(relation: "Relation", unknown: str) -> Callable | None:
    """Return numeric solver callable for the requested unknown."""
    from .registry import canonical_variable_name

    if relation.is_forward_only:
        return None
    unknown = canonical_variable_name(str(unknown))
    if unknown not in relation.symbols:
        return None
    if unknown in relation.outputs:
        return relation.forward
    if unknown in relation.solvers:
        return relation.solvers[unknown][1]
    if relation.sympy_expression is None:
        return None

    candidate_symbol = relation.symbols.get(unknown)
    if candidate_symbol is None:
        return None

    try:
        solutions = sp.solve(relation.sympy_expression, candidate_symbol)
    except Exception:
        return None
    if not solutions:
        return None

    ordered = tuple(name for name in relation.symbols if name != unknown)
    try:
        args = [relation.symbols[name] for name in ordered]
        solver = sp.lambdify(args, solutions[0], modules=["numpy", "sympy"])
    except Exception:
        return None

    solvers = dict(relation.solvers)
    solvers[unknown] = (ordered, solver)
    object.__setattr__(relation, "solvers", solvers)
    return solver


def solve_relation_for_value(
    relation: "Relation", unknown: str, values: Mapping[str, object]
) -> object | None:
    """Solve for one variable from a mapping of known values."""
    from .registry import canonical_variable_name

    if relation.is_forward_only:
        return None
    unknown = canonical_variable_name(str(unknown))
    if unknown not in relation.symbols:
        return None

    override_fn = relation.inverse_functions.get(unknown)
    if override_fn is not None:
        try:
            return override_fn(values)
        except Exception:
            return None

    if unknown in relation.outputs:
        try:
            return evaluate_relation(relation, values, target=unknown)
        except Exception:
            return None

    if unknown not in relation.solvers:
        if inverse_solver_for_relation(relation, unknown) is None:
            return None

    solver = relation.solvers.get(unknown)
    if solver is None:
        return None

    ordered, fn = solver
    try:
        args = tuple(values[name] for name in ordered)
    except KeyError:
        return None

    try:
        result = fn(*args)
    except Exception:
        return None

    try:
        scalar = float(result)
    except Exception:
        scalar = None
    return scalar if scalar is not None else result

def relation(
    *,
    name: str | None = None,
    output: str | None = None,
    outputs: Iterable[str] | None = None,
    inputs: Iterable[str] | None = None,
    tags: Iterable[str] | str | None = None,
    rel_tol_default: float | None = None,
    abs_tol_default: float | None = None,
    constraints: Iterable[str] | str | None = None,
    soft_constraints: Iterable[str] | str | None = None,
    initial_guesses: dict[str, Callable] | None = None,
    inverse_functions: dict[str, Callable] | None = None,
):
    """Decorator that builds and registers :class:`~fusdb.relation_class.Relation`.

    The decorated callable is replaced by a frozen :class:`~fusdb.relation_class.Relation`
    instance and appended to :data:`_RELATION_REGISTRY`.
    """

    from .relation_class import Relation

    def decorator(func: Callable) -> Relation:
        """Create one relation object from ``func`` and register it.
        Uses Relation.from_callable to build the object, which handles sympy parsing and validation.
        """
        if (output is None) == (outputs is None):
            raise ValueError("relation() requires exactly one of output= or outputs=")

        constraints_tuple = (
            ()
            if constraints is None
            else (constraints,) if isinstance(constraints, str) else tuple(constraints)
        )
        soft_constraints_tuple = (
            ()
            if soft_constraints is None
            else (soft_constraints,) if isinstance(soft_constraints, str) else tuple(soft_constraints)
        )

        relation_obj = Relation.from_callable(
            name=name or func.__name__,
            target=output,
            outputs=outputs,
            func=func,
            inputs=inputs,
            tags=normalize_tags_to_tuple(tags),
            rel_tol_default=rel_tol_default,
            abs_tol_default=abs_tol_default,
            constraints=constraints_tuple,
            soft_constraints=soft_constraints_tuple,
            initial_guesses=initial_guesses or {},
            inverse_functions=inverse_functions or {},
        )
        _RELATION_REGISTRY.append(relation_obj)
        return relation_obj

    return decorator


def get_filtered_relations(
    reactor_tags: Iterable[str] | str | None,
    variable_names: Iterable[str] | None,
    variable_methods: Iterable[str] | None,
    verbose: bool = False,
    extra_relations: Iterable[object] | None = None,
):
    """Get filtered relations based on tags and optional domain tags.

    Tag inputs are normalized internally (lowercase, alphanumeric).
    Unrecognized tags are simply ignored. This function loads all relation
    modules so the registry is populated; it does not require a Reactor object.

    Args:
        reactor_tags: Tags identifying the reactor type (any case/format)
        variable_names: Names of variables available in the system
        variable_methods: Relation names selected as method overrides
        verbose: If True, log warnings when tags are rejected
        extra_relations: Optional relations to include alongside the registry

    Returns:
        List of relations matching the criteria (or all if no filters)
    """
    from . import relations
    from .registry import (
        canonical_variable_name,
        load_allowed_tags,
    )
    from .utils import normalize_tags_to_tuple
    import logging

    logger = logging.getLogger(__name__)

    # Ensure relations are loaded before filtering (no Reactor instance required)
    relations.import_relations()

    allowed_tags = load_allowed_tags()
    solving_order = allowed_tags.get("solving_order", {}) or {}
    allowed_domains = set(solving_order)
    allowed_specific = (
        set(allowed_tags.get("reactor_configurations", ()) or ())
        | set(allowed_tags.get("confinement_modes", ()) or ())
        | set(allowed_tags.get("reactor_families", ()) or ())
    )

    # Normalize inputs
    tags_normalized = tuple(normalize_tags_to_tuple(reactor_tags or ()))
    variable_names_set = set(variable_names or ())
    method_names = {name for name in (variable_methods or ()) if name}
    domain_filters = {tag for tag in tags_normalized if tag in allowed_domains}
    reactor_tags_normalized = tuple(tag for tag in tags_normalized if tag not in allowed_domains)

    relations = list(_RELATION_REGISTRY)
    if extra_relations:
        seen = {id(rel) for rel in relations}
        for rel in extra_relations:
            if id(rel) in seen:
                continue
            relations.append(rel)
            seen.add(id(rel))

    # If no filtering, return all
    if not reactor_tags_normalized and not domain_filters and not variable_names_set:
        if verbose:
            logger.info("No filters applied; returning all %d relations", len(relations))
        return list(relations)

    results = []
    tags_set = set(reactor_tags_normalized)

    override_outputs: set[str] = set()
    if method_names:
        for relation in relations:
            if relation.name in method_names:
                for target in relation.outputs:
                    override_outputs.add(canonical_variable_name(target))

    for relation in relations:
        relation_tags = set(relation.tags)  # Already normalized by decorator
        domains = relation_tags.intersection(allowed_domains)
        reactor_specific = relation_tags.intersection(allowed_specific)

        # Domain filter (domain tags are part of reactor_tags)
        if domain_filters:
            if relation_tags.intersection(domain_filters):
                pass
            elif domains and not domains.intersection(domain_filters):
                if verbose:
                    logger.info("Rejecting %s: domain %s not in tags", relation.name, sorted(domain_filters))
                continue
            elif not domains and relation.name not in domain_filters:
                if verbose:
                    logger.info("Rejecting %s: no domain match for %s", relation.name, sorted(domain_filters))
                continue

        # Reactor tags must satisfy non-domain relation tags
        if reactor_specific and not reactor_specific.issubset(tags_set):
            if verbose:
                missing = reactor_specific - tags_set
                logger.info("Rejecting %s: missing tags %s", relation.name, sorted(missing))
            continue

        # Method override for outputs
        if override_outputs:
            relation_outputs = {
                canonical_variable_name(name)
                for name in relation.outputs
            }
            if relation_outputs.intersection(override_outputs) and relation.name not in method_names:
                if verbose:
                    logger.info("Rejecting %s: method override", relation.name)
                continue

        results.append(relation)

    if verbose:
        tags_label = ", ".join(reactor_tags_normalized) if reactor_tags_normalized else "none"
        domain_label = ", ".join(sorted(domain_filters)) if domain_filters else "none"
        logger.info(
            "Filtered relations: %d of %d (domain=%s, tags=%s)",
            len(results),
            len(_RELATION_REGISTRY),
            domain_label,
            tags_label,
        )
    return results
