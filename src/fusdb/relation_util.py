"""Shared utilities for relation definitions and solver constraint handling."""

import math
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.relational import Relational
import yaml

from fusdb.registry import VARIABLES_PATH

REL_TOL_DEFAULT = 1e-2  # relative tolerance for relation checks
_SYMBOLS: dict[str, sp.Symbol] = {}


def symbol(name: str) -> sp.Symbol:
    """Get a stable Sympy symbol for a variable name."""
    # Cache symbols to keep identity stable across relations.
    sym = _SYMBOLS.get(name)
    if sym is None:
        sym = sp.Symbol(name, real=True)
        _SYMBOLS[name] = sym
    return sym


@lru_cache(maxsize=1)
def _allowed_variables_raw() -> dict[str, object]:
    """Load the raw allowed variables mapping from YAML."""
    # Read the registry once and validate its shape.
    data = yaml.safe_load(VARIABLES_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("allowed_variables.yaml must contain a mapping")
    return data


@lru_cache(maxsize=1)
def _constraint_symbols() -> dict[str, sp.Symbol]:
    """Build a symbol table for allowed variables used in constraints."""
    # Create a shared symbol table so constraints parse consistently.
    return {name: symbol(name) for name in _allowed_variables_raw().keys()}


def parse_constraint(expr: str | Relational) -> Relational:
    """Parse a constraint string into a Sympy relational."""
    # Allow pre-parsed constraints to pass through.
    if isinstance(expr, Relational):
        return expr
    # Parse string constraints using the allowed-variable symbol table.
    if not isinstance(expr, str):
        raise ValueError(f"Constraint must be a string or Sympy relational; got {type(expr).__name__}")
    parsed = parse_expr(expr, local_dict=_constraint_symbols(), evaluate=False)
    # Ensure the parsed value is a relational expression.
    if isinstance(parsed, bool) or parsed in (sp.true, sp.false):
        raise ValueError(f"Constraint {expr!r} must be a relational expression")
    if not isinstance(parsed, Relational):
        raise ValueError(f"Constraint {expr!r} must be a relational expression")
    return parsed


@lru_cache(maxsize=1)
def variable_constraints() -> dict[str, tuple[Relational, ...]]:
    """Load per-variable constraint expressions from allowed variables YAML."""
    # Parse registry constraints once and cache them.
    constraints: dict[str, tuple[Relational, ...]] = {}
    data = _allowed_variables_raw()
    for name, meta in data.items():
        if not isinstance(meta, dict):
            continue
        raw = meta.get("constraints")
        if raw is None:
            continue
        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            raise ValueError(f"constraints for {name!r} must be a string or list of strings")
        parsed = tuple(parse_constraint(item) for item in items)
        constraints[name] = parsed
    return constraints


def constraints_for_vars(names: Iterable[str]) -> tuple[Relational, ...]:
    """Return the combined constraints for a collection of variable names."""
    # Collect all per-variable constraints for the given names.
    merged: list[Relational] = []
    mapping = variable_constraints()
    for name in names:
        merged.extend(mapping.get(name, ()))
    return tuple(merged)


def numeric_value(val: sp.Expr | float | int) -> float | None:
    """Return a finite float for numeric Sympy expressions or Python numbers."""
    # Evaluate Sympy expressions safely without introducing symbolic dependencies.
    if isinstance(val, sp.Expr):
        if val.free_symbols:
            return None
        try:
            evaluated = val.evalf(chop=True)
        except Exception:
            return None
        if evaluated.is_real is False:
            return None
        try:
            number = float(evaluated)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(val) else None
    return None


def as_float(value: Any) -> float | None:
    """Return a finite float for numeric values, else None."""
    # Filter out non-numeric and relational inputs early.
    if value is None or isinstance(value, Relational):
        return None
    if isinstance(value, sp.Expr):
        return numeric_value(value)
    if isinstance(value, (int, float)):
        return numeric_value(value)
    return None


def first_numeric(
    values: Mapping[str, Any],
    *names: str,
    default: float | None = None,
) -> float | None:
    """Return the first numeric value among the requested keys."""
    # Walk keys in order and return the first numeric value.
    for name in names:
        numeric = as_float(values.get(name))
        if numeric is not None:
            return numeric
    return default


def update_value(
    values: MutableMapping[str, float],
    name: str,
    new_value: float,
    *,
    eps: float = 1e-12,
) -> bool:
    """Update a numeric value if it differs beyond a relative epsilon."""
    # Use a relative threshold to avoid noisy updates.
    old_value = values.get(name)
    if old_value is None or abs(old_value - new_value) > eps * max(abs(old_value), abs(new_value), 1.0):
        values[name] = new_value
        return True
    return False


def relation_residual(
    values: Mapping[str, float],
    vars: Sequence[str],
    residual_fn: Callable[..., object] | None,
    expr: sp.Expr,
) -> float | None:
    """Evaluate a relation residual using lambdify when possible."""
    # Require all inputs before attempting evaluation.
    if any(name not in values for name in vars):
        return None
    # Prefer the lambdified residual for speed.
    args = [values[name] for name in vars]
    if residual_fn is not None:
        try:
            result = residual_fn(*args)
        except Exception:
            result = None
        else:
            if isinstance(result, (list, tuple)) and result:
                result = result[0]
            try:
                numeric = float(result)
            except (TypeError, ValueError):
                numeric = None
            else:
                if math.isfinite(numeric):
                    return numeric
    # Fallback to symbolic substitution when lambdify fails.
    subs = {symbol(name): sp.Float(values[name]) for name in vars}
    return numeric_value(expr.subs(subs))


def solve_linear_system(
    equations: Sequence[sp.Expr],
    unknowns: Sequence[sp.Symbol],
) -> dict[sp.Symbol, sp.Expr] | None:
    """Solve a linear system if possible, returning a symbol->expr map."""
    # Convert to a matrix form and attempt a linear solve.
    try:
        matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)
        solutions = sp.linsolve((matrix, rhs), unknowns)
    except Exception:
        return None
    for sol in solutions:
        return dict(zip(unknowns, sol))
    return None


def solve_numeric_system(
    equations: Sequence[sp.Expr],
    unknowns: Sequence[sp.Symbol],
    guesses: Sequence[float],
    *,
    max_iter: int,
) -> list[float] | None:
    """Solve a nonlinear system numerically, preferring least_squares."""
    # Try least_squares first when SciPy is available.
    try:
        from scipy.optimize import least_squares  # type: ignore[import-not-found]
    except Exception:
        least_squares = None

    if least_squares is not None:
        func = sp.lambdify(unknowns, equations, "math")

        # Map vector inputs to residual list.
        def residuals(x: Sequence[float]) -> list[float]:
            result = func(*x)
            if isinstance(result, (list, tuple)):
                return [float(val) for val in result]
            return [float(result)]

        try:
            lsq = least_squares(residuals, guesses, max_nfev=max_iter * 20)
        except Exception:
            lsq = None
        if lsq is not None and lsq.success:
            return [float(val) for val in lsq.x]

    # Fall back to nsolve when the system is square.
    if len(equations) != len(unknowns):
        return None
    try:
        nsolve_solution = sp.nsolve(equations, unknowns, guesses, tol=1e-10, maxsteps=max_iter)
    except Exception:
        return None
    if len(unknowns) == 1:
        return [float(nsolve_solution)]
    try:
        return [float(val) for val in nsolve_solution]
    except TypeError:
        return [float(nsolve_solution)]


def constraints_ok(
    constraints: Sequence[tuple[tuple[str, ...], Callable[..., object] | None, Relational]],
    values: Mapping[str, float],
    *,
    focus_names: set[str] | None = None,
) -> bool:
    """Evaluate precompiled constraints against known values."""
    if not constraints:
        return True
    # Optionally restrict checks to constraints that touch updated variables.
    focus = focus_names or set()
    for names, fn, constraint in constraints:
        if focus and names and not (set(names) & focus):
            continue
        if any(name not in values for name in names):
            continue
        args = [values[name] for name in names]
        verdict = None
        if fn is not None:
            # Prefer fast lambdified constraint evaluation.
            try:
                verdict = fn(*args)
            except Exception:
                verdict = None
            else:
                if isinstance(verdict, bool):
                    if verdict is False:
                        return False
                    continue
                if verdict is True or verdict == sp.true:
                    continue
                if verdict is False or verdict == sp.false:
                    return False
                verdict = None

        # Fallback to symbolic substitution when lambdify is inconclusive.
        if verdict is None:
            subs = {symbol(name): sp.Float(values[name]) for name in names}
            try:
                evaluated = constraint.subs(subs)
            except Exception:
                continue
            if evaluated is True or evaluated == sp.true:
                continue
            if evaluated is False or evaluated == sp.false:
                return False
            try:
                if bool(evaluated) is False:
                    return False
            except Exception:
                continue
    return True
