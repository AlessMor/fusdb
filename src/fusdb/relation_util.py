"""Utilities shared by :class:`fusdb.relation_class.Relation` objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping
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

def relation(
    *,
    name: str | None = None,
    output: str,
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
            func=func,
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
        ALLOWED_CONFINEMENT_MODES,
        ALLOWED_REACTOR_CONFIGURATIONS,
        ALLOWED_REACTOR_FAMILIES,
        ALLOWED_SOLVING_ORDER,
        canonical_variable_name,
    )
    from .utils import normalize_tags_to_tuple
    import logging

    logger = logging.getLogger(__name__)

    # Ensure relations are loaded before filtering (no Reactor instance required)
    relations.import_relations()

    # Normalize inputs
    tags_normalized = tuple(normalize_tags_to_tuple(reactor_tags or ()))
    variable_names_set = set(variable_names or ())
    method_names = {name for name in (variable_methods or ()) if name}

    allowed_domains = set(ALLOWED_SOLVING_ORDER)
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
    allowed_specific = (
        set(ALLOWED_REACTOR_CONFIGURATIONS)
        | set(ALLOWED_CONFINEMENT_MODES)
        | set(ALLOWED_REACTOR_FAMILIES)
    )
    tags_set = set(reactor_tags_normalized)

    override_outputs: set[str] = set()
    if method_names:
        for relation in relations:
            if relation.name in method_names:
                target = (relation._preferred_target if relation._preferred_target is not None else next(iter(relation.numeric_functions), None))
                if target:
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
            target = (relation._preferred_target if relation._preferred_target is not None else next(iter(relation.numeric_functions), None))
            if target:
                output_name = canonical_variable_name(target)
                if output_name in override_outputs and relation.name not in method_names:
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
