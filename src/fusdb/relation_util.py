"""Utilities shared by Relation objects."""

from __future__ import annotations

from typing import Callable, Iterable
import types
import inspect

import sympy as sp


def function_inputs(func: Callable) -> list[str]:
    """Return ordered input names for a callable.

    Args:
        func: Callable to introspect.

    Returns:
        Ordered list of parameter names.
    """
    sig = inspect.signature(func)
    inputs: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            inputs.append(name)
    return inputs


def build_sympy_expr(
    func: Callable,
    inputs: Iterable[str],
    output: str,
) -> tuple[sp.Expr | None, dict[str, sp.Symbol] | None]:
    """Create an implicit sympy expression output - f(inputs).

    Args:
        func: Explicit relation function.
        inputs: Ordered input variable names.
        output: Output variable name.

    Returns:
        (expr, symbols) or (None, None) if conversion fails.
    """
    symbols = {name: sp.Symbol(name, real=True) for name in (*inputs, output)}
    try:
        expr = func(*[symbols[name] for name in inputs])
        return symbols[output] - expr, symbols
    except Exception:
        pass

    class _SympyModuleProxy:
        """Proxy math/numpy-style functions to sympy equivalents (best-effort)."""

        _ALT = {
            "arcsin": "asin",
            "arccos": "acos",
            "arctan": "atan",
            "arcsinh": "asinh",
            "arccosh": "acosh",
            "arctanh": "atanh",
            "log10": "log",
            "log2": "log",
            "power": "Pow",
            "abs": "Abs",
            "fabs": "Abs",
        }

        _CONST = {
            "pi": sp.pi,
            "e": sp.E,
        }

        def __getattr__(self, name: str) -> object:
            if name in self._CONST:
                return self._CONST[name]
            if hasattr(sp, name):
                return getattr(sp, name)
            alt = self._ALT.get(name)
            if alt and hasattr(sp, alt):
                return getattr(sp, alt)
            raise AttributeError(name)

    proxy = _SympyModuleProxy()
    patched_globals = dict(func.__globals__)
    patched_globals.setdefault("__builtins__", func.__globals__.get("__builtins__", __builtins__))
    # Swap common module handles if present.
    if "math" in patched_globals:
        patched_globals["math"] = proxy
    if "np" in patched_globals:
        patched_globals["np"] = proxy
    if "numpy" in patched_globals:
        patched_globals["numpy"] = proxy
    # Swap common direct imports (sin, cos, pi, etc.) if present.
    for name in (
        "pi",
        "e",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "sqrt",
        "exp",
        "log",
        "log10",
        "log2",
        "power",
        "abs",
        "fabs",
        "floor",
        "ceil",
    ):
        if name in patched_globals:
            try:
                patched_globals[name] = getattr(proxy, name)
            except AttributeError:
                continue
    try:
        patched_func = types.FunctionType(
            func.__code__,
            patched_globals,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        expr = patched_func(*[symbols[name] for name in inputs])
        return symbols[output] - expr, symbols
    except Exception:
        return None, None


def evaluate_constraints(
    constraints: Iterable[str] | None,
    values: dict[str, object],
) -> list[str]:
    """Evaluate constraint strings against provided values.

    Args:
        constraints: Iterable of constraint expressions.
        values: Mapping of variable names to numeric values.

    Returns:
        List of violated constraint strings (empty if none).
    """
    if not constraints:
        return []
    failed: list[str] = []
    for constraint in constraints:
        try:
            expr = sp.sympify(constraint, locals=values)
            if expr is sp.S.true:
                continue
            if expr is sp.S.false:
                failed.append(constraint)
                continue
            if not bool(expr):
                failed.append(constraint)
        except Exception:
            # If evaluation fails, keep it undecidable rather than failing hard.
            continue
    return failed


def check_relation_satisfied(
    rel: object,
    values: dict[str, object],
    *,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    check_constraints: bool = True
) -> tuple[bool, str, float | None]:
    """Check if relation is satisfied with given values.
    
    Performs a comprehensive check including:
    1. Verifying all variables are present
    2. Checking constraints (if requested)
    3. Evaluating the relation
    4. Computing residual (actual - expected)
    5. Checking if within tolerance
    
    Args:
        rel: Relation object to check
        values: Variable values dictionary
        rel_tol: Relative tolerance (uses rel.rel_tol_default if None)
        abs_tol: Absolute tolerance (uses rel.abs_tol_default if None)
        check_constraints: If True, check constraints before evaluating
        
    Returns:
        Tuple of (satisfied, status, residual) where:
        - satisfied: True if relation is satisfied within tolerance
        - status: "SAT" (satisfied), "VIOLATED" (outside tolerance), or "UNDECIDABLE" (cannot evaluate)
        - residual: actual - expected value, or None if undecidable
    """
    from .utils import within_tolerance
    
    # Check if all variables are present
    if any(name not in values for name in rel.variables):
        return (False, "UNDECIDABLE", None)
    
    # Check constraints if requested
    if check_constraints:
        violations = evaluate_constraints(rel.constraints, values)
        if violations:
            return (False, "VIOLATED", None)
    
    # Evaluate relation
    try:
        expected = rel.evaluate(**{name: values[name] for name in rel.inputs})
    except Exception:
        return (False, "UNDECIDABLE", None)
    
    actual = values.get(rel.output)
    
    # Convert to scalars
    try:
        actual_scalar = float(actual) if actual is not None else None
        expected_scalar = float(expected) if expected is not None else None
    except Exception:
        return (False, "UNDECIDABLE", None)
    
    if actual_scalar is None or expected_scalar is None:
        return (False, "UNDECIDABLE", None)
    
    # Calculate residual
    residual = actual_scalar - expected_scalar
    
    # Check tolerance
    rel_tol_val = rel_tol if rel_tol is not None else (rel.rel_tol_default or 0.0)
    abs_tol_val = abs_tol if abs_tol is not None else (rel.abs_tol_default or 0.0)
    
    satisfied = within_tolerance(actual_scalar, expected_scalar, 
                                 rel_tol=rel_tol_val, abs_tol=abs_tol_val)
    status = "SAT" if satisfied else "VIOLATED"
    
    return (satisfied, status, residual)


def build_inverse_solver(
    expr: sp.Expr,
    symbols: dict[str, sp.Symbol],
    unknown: str,
    ordered_vars: Iterable[str],
) -> Callable | None:
    """Try to build a numeric inverse solver for a single unknown.

    Args:
        expr: Implicit expression equal to 0.
        symbols: Sympy symbols by name.
        unknown: Variable name to solve for.
        ordered_vars: All variable names in order.

    Returns:
        Callable or None if an inverse cannot be built.
    """
    try:
        solutions = sp.solve(expr, symbols[unknown])
    except Exception:
        return None
    if not solutions:
        return None
    try:
        args = [symbols[name] for name in ordered_vars if name != unknown]
        return sp.lambdify(args, solutions[0], modules=["numpy", "sympy"])
    except Exception:
        return None


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
    from .relation_class import _RELATION_REGISTRY
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
                override_outputs.add(canonical_variable_name(relation.output))

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
            output_name = canonical_variable_name(relation.output)
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
