"""
Relation utilities: parsing, constraint handling, numeric solving.

This module provides the low-level utilities for working with mathematical
relations in fusdb. It handles:
- Sympy symbol management (consistent symbols across the codebase)
- Constraint parsing (converting strings like "x >= 0" to Sympy relationals)
- Constraint evaluation (fast checking if constraints are satisfied)
- Numeric value extraction (safely converting Sympy expressions to floats)
- Equation solving (both symbolic and numeric methods)

These utilities are used by relation_class.py to implement the higher-level
RelationSystem solver.
"""
from __future__ import annotations
import math
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.relational import Relational
import yaml
from fusdb.registry import VARIABLES_PATH

# Default relative tolerance for considering equations satisfied
REL_TOL_DEFAULT = 1e-2

# Cache for Sympy symbols (ensures same variable name always maps to same symbol)
_SYMBOLS: dict[str, sp.Symbol] = {}


def symbol(name: str) -> sp.Symbol:
    """
    Get or create a stable Sympy symbol for a variable name.
    
    This function ensures that the same variable name always returns
    the same Sympy Symbol object, which is important for symbolic
    computations where symbol identity matters.
    
    Args:
        name: The variable name (e.g., "P_fus", "R", "a")
        
    Returns:
        A Sympy Symbol with the given name, marked as real-valued
        
    Example:
        >>> s1 = symbol("x")
        >>> s2 = symbol("x")
        >>> s1 is s2  # Same object
        True
    """
    if name not in _SYMBOLS:
        _SYMBOLS[name] = sp.Symbol(name, real=True)
    return _SYMBOLS[name]


@lru_cache(maxsize=1)
def _load_variables_yaml() -> dict[str, object]:
    """
    Load the allowed_variables.yaml file (cached).
    
    This file defines all valid variable names and their metadata
    including units, descriptions, and per-variable constraints.
    
    Returns:
        Dictionary mapping variable names to their metadata
        
    Raises:
        ValueError: If the file doesn't contain a valid mapping
    """
    data = yaml.safe_load(VARIABLES_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("allowed_variables.yaml must contain a mapping")
    return data


@lru_cache(maxsize=1)
def _constraint_symbol_table() -> dict[str, sp.Symbol]:
    """
    Build a symbol table for parsing constraint expressions.
    
    Creates a mapping from all known variable names to their
    Sympy symbols, used by the expression parser.
    
    Returns:
        Dictionary mapping variable names to Sympy symbols
    """
    return {name: symbol(name) for name in _load_variables_yaml()}


def parse_constraint(expr: str | Relational) -> Relational:
    """
    Parse a constraint string into a Sympy relational.
    
    Converts string expressions like "P >= 0" or "0 <= f_GW <= 1"
    into Sympy Relational objects that can be evaluated.
    
    Args:
        expr: Either a string constraint or an existing Relational
        
    Returns:
        A Sympy Relational object representing the constraint
        
    Raises:
        ValueError: If the string doesn't represent a valid constraint
        
    Example:
        >>> con = parse_constraint("P >= 0")
        >>> type(con)
        <class 'sympy.core.relational.GreaterThan'>
    """
    if isinstance(expr, Relational):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"Constraint must be str or Relational; got {type(expr).__name__}")
    
    # Parse the expression using the known variable symbols
    parsed = parse_expr(expr, local_dict=_constraint_symbol_table(), evaluate=False)
    
    # Verify it's a relational expression (not just a boolean or number)
    if isinstance(parsed, bool) or parsed in (sp.true, sp.false) or not isinstance(parsed, Relational):
        raise ValueError(f"Constraint {expr!r} must be a relational expression")
    return parsed


@lru_cache(maxsize=1)
def variable_constraints() -> dict[str, tuple[Relational, ...]]:
    """
    Load per-variable constraints from allowed_variables.yaml.
    
    Each variable can have constraints defined in the YAML file,
    such as "P >= 0" for non-negative power. This function loads
    and parses all such constraints.
    
    Returns:
        Dictionary mapping variable names to tuples of their constraints
        
    Example:
        >>> vc = variable_constraints()
        >>> vc.get("f_GW")  # Greenwald fraction: 0 <= f_GW <= 1
        (f_GW >= 0, f_GW <= 1)
    """
    constraints: dict[str, tuple[Relational, ...]] = {}
    for name, meta in _load_variables_yaml().items():
        if not isinstance(meta, dict):
            continue
        raw = meta.get("constraints")
        if raw is None:
            continue
        # Handle both single constraint and list of constraints
        items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else None
        if items is None:
            raise ValueError(f"constraints for {name!r} must be str or list")
        constraints[name] = tuple(parse_constraint(item) for item in items)
    return constraints


def constraints_for_vars(names: Iterable[str]) -> tuple[Relational, ...]:
    """
    Get all constraints that apply to a set of variables.
    
    Looks up the per-variable constraints for each variable name
    and returns them all as a single tuple.
    
    Args:
        names: Iterable of variable names
        
    Returns:
        Tuple of all constraints that apply to any of the variables
    """
    mapping = variable_constraints()
    return tuple(c for name in names for c in mapping.get(name, ()))


def compile_constraints(
    constraints: Sequence[Relational | str],
    variables: Iterable[str]
) -> tuple[tuple[tuple[str, ...], Callable[..., object] | None, Relational], ...]:
    """
    Compile constraints for fast evaluation.
    
    Takes a sequence of constraints (either strings or Relationals)
    and compiles them into a format optimized for fast evaluation.
    Also includes per-variable constraints from allowed_variables.yaml.
    
    Args:
        constraints: Sequence of constraint expressions
        variables: Variable names involved (to include their per-variable constraints)
        
    Returns:
        Tuple of compiled constraints, each being:
        - names: tuple of variable names the constraint depends on
        - fn: lambdified function for fast evaluation (or None)
        - con: original Relational object
        
    Example:
        >>> compiled = compile_constraints(["P >= 0"], ["P", "V", "I"])
        >>> names, fn, con = compiled[0]
        >>> fn(100)  # Evaluate P >= 0 with P=100
        True
    """
    compiled = []
    # Include both explicit constraints and per-variable constraints
    all_constraints = list(constraints) + list(constraints_for_vars(variables))
    
    for item in all_constraints:
        # Parse if string
        con = item if isinstance(item, Relational) else parse_constraint(item)
        
        # Get variable names the constraint depends on
        names = tuple(sorted(sym.name for sym in con.free_symbols))
        
        # Try to compile a fast evaluation function
        try:
            fn = sp.lambdify([symbol(n) for n in names], con, "math")
        except Exception:
            fn = None
        
        compiled.append((names, fn, con))
    
    return tuple(compiled)


def numeric_value(val: Any) -> float | None:
    """
    Extract a finite float from a numeric value; None if not numeric or non-finite.
    
    Safely converts various numeric types (int, float, Sympy expressions)
    to Python floats, returning None if the value is not numeric,
    contains free symbols, or is not finite (inf, nan).
    
    Args:
        val: Any value to convert
        
    Returns:
        The numeric value as a float, or None if conversion fails
        
    Example:
        >>> numeric_value(42)
        42.0
        >>> numeric_value(sp.sqrt(4))
        2.0
        >>> numeric_value(sp.Symbol("x"))  # Has free symbols
        None
        >>> numeric_value(float("inf"))
        None
    """
    if val is None or isinstance(val, Relational):
        return None
    
    # Handle Sympy expressions
    if isinstance(val, sp.Expr):
        if val.free_symbols:  # Can't evaluate if has unknowns
            return None
        try:
            n = float(val.evalf(chop=True))
            return n if math.isfinite(n) else None
        except Exception:
            return None
    
    # Handle Python numeric types
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(val) else None
    
    return None


# Alias for backward compatibility
as_float = numeric_value


def first_numeric(values: Mapping[str, Any], *names: str, default: float | None = None) -> float | None:
    """
    Get the first numeric value found for any of the given names.
    
    Searches through the provided names in order and returns the
    first one that has a valid numeric value.
    
    Args:
        values: Dictionary of values to search
        *names: Variable names to try, in order of preference
        default: Value to return if none found
        
    Returns:
        The first numeric value found, or default if none
        
    Example:
        >>> vals = {"a": None, "b": 10.0, "c": 20.0}
        >>> first_numeric(vals, "a", "b", "c")  # "a" is None, returns "b"
        10.0
    """
    for name in names:
        n = numeric_value(values.get(name))
        if n is not None:
            return n
    return default


def update_value(values: MutableMapping[str, float], name: str, new_value: float, *, eps: float = 1e-12) -> bool:
    """
    Update a value in the dictionary if it's significantly different.
    
    Only updates if the new value differs from the old by more than
    eps times the magnitude (avoids unnecessary updates for tiny changes).
    
    Args:
        values: Dictionary to update
        name: Key to update
        new_value: New value to set
        eps: Relative tolerance for considering values different
        
    Returns:
        True if the value was updated, False if unchanged
    """
    old = values.get(name)
    if old is None or abs(old - new_value) > eps * max(abs(old), abs(new_value), 1.0):
        values[name] = new_value
        return True
    return False


def relation_residual(
    values: Mapping[str, float],
    variables: Sequence[str],
    residual_fn: Callable[..., object] | None,
    expr: sp.Expr
) -> float | None:
    """
    Evaluate the residual of a relation (how far from being satisfied).
    
    For a relation f(x₁, x₂, ...) = 0, the residual is the value of f
    at the given point. A residual of 0 means the relation is exactly satisfied.
    
    Args:
        values: Dictionary of variable values
        variables: Ordered list of variable names
        residual_fn: Pre-compiled function for fast evaluation (may be None)
        expr: Sympy expression for symbolic evaluation fallback
        
    Returns:
        The residual value, or None if any variable is missing
        
    Example:
        >>> # For relation P - V*I = 0 with P=100, V=10, I=10
        >>> relation_residual({"P": 100, "V": 10, "I": 10}, ["P", "V", "I"], fn, expr)
        0.0  # Satisfied
        >>> relation_residual({"P": 100, "V": 10, "I": 9}, ["P", "V", "I"], fn, expr)
        10.0  # Not satisfied
    """
    # Check all variables are available
    if any(n not in values for n in variables):
        return None
    
    args = [values[n] for n in variables]
    
    # Try fast compiled function first
    if residual_fn:
        try:
            r = residual_fn(*args)
            # Handle case where function returns a list/tuple
            if isinstance(r, (list, tuple)) and r:
                r = r[0]
            n = float(r)
            if math.isfinite(n):
                return n
        except Exception:
            pass
    
    # Fall back to symbolic evaluation
    subs = {symbol(n): sp.Float(values[n]) for n in variables}
    return numeric_value(expr.subs(subs))


def _eval_result(v: Any) -> bool | None:
    """
    Convert a value to a boolean constraint result.
    
    Handles both Python bools and Sympy booleans.
    
    Args:
        v: Value to convert
        
    Returns:
        True, False, or None (if indeterminate)
    """
    if v is True or v == sp.true:
        return True
    if v is False or v == sp.false:
        return False
    if isinstance(v, bool):
        return v
    return None


def constraints_ok(
    constraints: Sequence[tuple[tuple[str, ...], Callable[..., object] | None, Relational]],
    values: Mapping[str, float],
    *,
    focus_names: set[str] | None = None
) -> bool:
    """
    Check if all constraints are satisfied by the given values.
    
    Evaluates each compiled constraint against the provided values,
    optionally focusing only on constraints involving specific variables.
    
    Args:
        constraints: Compiled constraints (from compile_constraints)
        values: Dictionary of variable values to test
        focus_names: If provided, only check constraints involving these variables
        
    Returns:
        True if all (focused) constraints are satisfied
        
    Example:
        >>> compiled = compile_constraints(["x >= 0", "y <= 10"], ["x", "y"])
        >>> constraints_ok(compiled, {"x": 5, "y": 3})
        True
        >>> constraints_ok(compiled, {"x": -1, "y": 3})
        False
    """
    if not constraints:
        return True
    
    focus = focus_names or set()
    
    for names, fn, con in constraints:
        # If focusing, skip constraints that don't involve focus variables
        if focus and names and not (set(names) & focus):
            continue
        
        # Skip if not all required variables are available
        if any(n not in values for n in names):
            continue
        
        args = [values[n] for n in names]
        
        # Try fast evaluation first
        if fn:
            try:
                r = _eval_result(fn(*args))
                if r is False:
                    return False
                if r is True:
                    continue
            except Exception:
                pass
        
        # Fall back to symbolic evaluation
        subs = {symbol(n): sp.Float(values[n]) for n in names}
        try:
            ev = con.subs(subs)
            r = _eval_result(ev)
            if r is False:
                return False
            if r is True:
                continue
            if bool(ev) is False:
                return False
        except Exception:
            continue
    
    return True


def solve_linear_system(
    equations: Sequence[sp.Expr],
    unknowns: Sequence[sp.Symbol]
) -> dict[sp.Symbol, sp.Expr] | None:
    """
    Solve a system of linear equations symbolically.
    
    Attempts to solve the system using Sympy's linear algebra facilities.
    This is fast and exact when the system is truly linear.
    
    Args:
        equations: List of expressions that should equal zero
        unknowns: List of symbols to solve for
        
    Returns:
        Dictionary mapping symbols to their solutions, or None if failed
        
    Example:
        >>> x, y = sp.symbols("x y")
        >>> solve_linear_system([x + y - 3, x - y - 1], [x, y])
        {x: 2, y: 1}
    """
    try:
        m, r = sp.linear_eq_to_matrix(equations, unknowns)
        for sol in sp.linsolve((m, r), unknowns):
            return dict(zip(unknowns, sol))
    except Exception:
        pass
    return None


def solve_numeric_system(
    equations: Sequence[sp.Expr],
    unknowns: Sequence[sp.Symbol],
    guesses: Sequence[float],
    *,
    max_iter: int
) -> list[float] | None:
    """
    Solve a system of equations numerically.
    
    Uses scipy's least_squares for optimization, falling back to
    sympy's nsolve for systems that don't converge well.
    
    Args:
        equations: List of expressions that should equal zero
        unknowns: List of symbols to solve for
        guesses: Initial guess values (one per unknown)
        max_iter: Maximum number of iterations
        
    Returns:
        List of solution values, or None if failed
        
    Example:
        >>> x = sp.Symbol("x")
        >>> solve_numeric_system([x**2 - 4], [x], [1.5], max_iter=50)
        [2.0]
    """
    try:
        from scipy.optimize import least_squares
        
        # Compile equations to a function
        func = sp.lambdify(unknowns, equations, "math")
        
        def res(x):
            r = func(*x)
            return [float(v) for v in r] if isinstance(r, (list, tuple)) else [float(r)]
        
        lsq = least_squares(res, guesses, max_nfev=max_iter * 20)
        if lsq.success:
            return [float(v) for v in lsq.x]
    except Exception:
        pass
    
    # Fall back to sympy's nsolve for square systems
    if len(equations) == len(unknowns):
        try:
            sol = sp.nsolve(equations, unknowns, guesses, tol=1e-10, maxsteps=max_iter)
            return [float(sol)] if len(unknowns) == 1 else [float(v) for v in sol]
        except Exception:
            pass
    
    return None


def solve_for_variable(
    expr: sp.Expr,
    target: str,
    values: Mapping[str, float],
    variables: Sequence[str],
    constraints_compiled: Sequence[tuple[tuple[str, ...], Callable[..., object] | None, Relational]],
    *,
    check_constraints: bool = True
) -> float | None:
    """
    Solve an expression for a single target variable symbolically.
    
    Substitutes known values into the expression and solves algebraically
    for the target variable. Optionally checks that the solution satisfies
    constraints.
    
    Args:
        expr: The expression (should equal zero)
        target: Name of the variable to solve for
        values: Dictionary of known variable values
        variables: All variable names in the expression
        constraints_compiled: Compiled constraints to check against
        check_constraints: If True, verify solution satisfies constraints
        
    Returns:
        The computed value, or None if no valid solution found
        
    Example:
        >>> # Solve P - V*I = 0 for P given V=10, I=5
        >>> P, V, I = sp.symbols("P V I")
        >>> solve_for_variable(P - V*I, "P", {"V": 10, "I": 5}, ["P", "V", "I"], [])
        50.0
    """
    tsym = symbol(target)
    
    # Substitute known values
    ksubs = {symbol(v): sp.Float(values[v]) for v in variables if v != target and v in values}
    sub = expr.subs(ksubs)
    
    # Check that only target variable remains
    if sub.free_symbols not in ({tsym}, set()):
        return None
    
    try:
        # Solve symbolically
        sols = sp.solve(sub, tsym)
        if not isinstance(sols, list):
            sols = [sols]
        
        for sol in sols:
            # Skip solutions with remaining free symbols
            if sol.free_symbols:
                continue
            
            n = numeric_value(sol)
            if n is None or not math.isfinite(n):
                continue
            
            # Check constraints if requested
            if check_constraints:
                t = dict(values)
                t[target] = n
                if not constraints_ok(constraints_compiled, t, focus_names={target}):
                    continue
            
            # Prefer positive solutions (common in physics)
            if n > 0:
                return n
    except Exception:
        pass
    
    return None


def extract_constraint_bounds(
    constraints_compiled: Sequence[tuple[tuple[str, ...], Callable[..., object] | None, Relational]],
    target: str
) -> tuple[float, float]:
    """
    Extract numeric bounds for a variable from its constraints.
    
    Parses constraints like "x >= 0" and "x <= 10" to determine
    the valid range [lo, hi] for a variable.
    
    Args:
        constraints_compiled: Compiled constraints
        target: Variable name to extract bounds for
        
    Returns:
        Tuple (lo, hi) with lower and upper bounds
        (uses -1e10 and 1e10 as defaults)
        
    Example:
        >>> compiled = compile_constraints(["x >= 0", "x <= 1"], ["x"])
        >>> extract_constraint_bounds(compiled, "x")
        (0.0, 1.0)
    """
    lo, hi = -1e10, 1e10
    
    for names, fn, con in constraints_compiled:
        # Only consider single-variable constraints on target
        if target not in names or len(names) != 1:
            continue
        
        s = str(con)
        
        # Parse <= and >= operators
        for op, is_lo in [('<=', False), ('>=', True)]:
            if op not in s:
                continue
            parts = s.split(op)
            if len(parts) != 2:
                continue
            try:
                # Determine which side is the bound
                b = float(parts[1].strip() if target in parts[0] else parts[0].strip())
                in_lhs = target in parts[0]
                
                # Update bounds based on operator and variable position
                if op == '<=' and in_lhs:
                    hi = min(hi, b)  # x <= b
                elif op == '<=' and not in_lhs:
                    lo = max(lo, b)  # b <= x
                elif op == '>=' and in_lhs:
                    lo = max(lo, b)  # x >= b
                elif op == '>=' and not in_lhs:
                    hi = min(hi, b)  # b >= x
            except Exception:
                pass
    
    return lo, hi
