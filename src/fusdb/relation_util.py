import math
from functools import lru_cache
from typing import Any, Callable, Iterable

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.relational import Relational
import yaml

from fusdb.registry import VARIABLES_PATH

WarnFunc = Callable[[str, type[Warning] | None], None]
REL_TOL_DEFAULT = 1e-2  # relative tolerance for relation checks
_SYMBOLS: dict[str, sp.Symbol] = {}


def symbol(name: str) -> sp.Symbol:
    """Get a stable Sympy symbol for a variable name."""
    sym = _SYMBOLS.get(name)
    if sym is None:
        sym = sp.Symbol(name, real=True)
        _SYMBOLS[name] = sym
    return sym


def nonzero(expr: sp.Expr) -> Relational:
    """Return a nonzero constraint for the expression."""
    return sp.Ne(expr, 0)


def positive(expr: sp.Expr) -> Relational:
    """Return a positive constraint for the expression."""
    return sp.Gt(expr, 0)


@lru_cache(maxsize=1)
def _allowed_variables_raw() -> dict[str, object]:
    """Load the raw allowed variables mapping from YAML."""
    data = yaml.safe_load(VARIABLES_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("allowed_variables.yaml must contain a mapping")
    return data


@lru_cache(maxsize=1)
def _constraint_symbols() -> dict[str, sp.Symbol]:
    """Build a symbol table for allowed variables used in constraints."""
    return {name: symbol(name) for name in _allowed_variables_raw().keys()}


def parse_constraint(expr: str | Relational) -> Relational:
    """Parse a constraint string into a Sympy relational."""
    if isinstance(expr, Relational):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"Constraint must be a string or Sympy relational; got {type(expr).__name__}")
    parsed = parse_expr(expr, local_dict=_constraint_symbols(), evaluate=False)
    if isinstance(parsed, bool) or parsed in (sp.true, sp.false):
        raise ValueError(f"Constraint {expr!r} must be a relational expression")
    if not isinstance(parsed, Relational):
        raise ValueError(f"Constraint {expr!r} must be a relational expression")
    return parsed


@lru_cache(maxsize=1)
def variable_constraints() -> dict[str, tuple[Relational, ...]]:
    """Load per-variable constraint expressions from allowed variables YAML."""
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
    merged: list[Relational] = []
    mapping = variable_constraints()
    for name in names:
        merged.extend(mapping.get(name, ()))
    return tuple(merged)


def require_nonzero(value: float | sp.Expr, field_name: str, context: str = "relation checks") -> float | sp.Expr:
    """Validate that a value is non-zero (numeric) or return it for symbolic usage.
    
    Args:
        value: The numeric value to check.
        field_name: The name of the field being validated.
        context: A description of the context where this check is performed.
        
    Raises:
        ValueError: If the value is zero.
    """
    if isinstance(value, sp.Expr):
        return value
    if value == 0:
        raise ValueError(f"{field_name} must be non-zero for {context}")
    return value


def coerce_number(value: Any, field_name: str) -> sp.Expr | None:
    """Convert a value to a Sympy expression, ensuring numeric inputs are finite.
    
    Args:
        value: The value to convert. Can be int, float, or None.
        field_name: The name of the field for error reporting.
        
    Returns:
        The value as a Sympy expression, or None if the input is None.
        
    Raises:
        ValueError: If the value is not numeric, not finite, or cannot be converted.
    """
    if value is None:
        return None
    if isinstance(value, sp.Expr):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be finite for relation checks")
        return sp.Float(number)
    raise ValueError(f"{field_name} must be numeric for relation checks; got {value!r}")
