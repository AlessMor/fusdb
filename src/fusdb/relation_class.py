from __future__ import annotations

import math
import operator
import warnings
from typing import Callable, Mapping, Sequence

import sympy as sp
from sympy.core.relational import Relational

from fusdb.relation_util import (
    REL_TOL_DEFAULT,
    coerce_number,
    constraints_for_vars,
    parse_constraint,
    symbol,
)

WarnFunc = Callable[[str, type[Warning] | None], None]

PRIORITY_EXPLICIT = 100
PRIORITY_RELATION = 10
PRIORITY_STRICT = 120

_REL_OPS: dict[str, Callable[[float, float], bool]] = {
    "==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
}


class Relation:
    """Implicit relation defined by a zero-residual Sympy expression."""

    def __init__(
        self,
        name: str,
        variables: tuple[str, ...],
        expr: sp.Expr,
        priority: int | None = None,
        rel_tol: float = REL_TOL_DEFAULT,
        solve_for: tuple[str, ...] | None = None,
        initial_guesses: Mapping[str, float | Callable[[Mapping[str, float]], float]] | None = None,
        max_solve_iterations: int = 25,
        constraints: Sequence[Relational | str] | None = None,
    ) -> None:
        """Store the metadata and symbolic form for a relation."""
        self.name = name
        self.variables = variables
        self.expr = sp.sympify(expr)
        self.priority = priority
        self.rel_tol = rel_tol
        self.solve_for = solve_for
        self.initial_guesses = initial_guesses
        self.max_solve_iterations = max_solve_iterations
        self.constraints = tuple(constraints or ())


class RelationSystem:
    """Iteratively solve simple relation chains without full global Sympy solving.

    Each relation is treated as an implicit equation with a preferred output
    variable (the first in relation.variables). The solver walks relations,
    substitutes known numeric values, and tries to solve for a single missing
    variable at a time (linear extraction first, then Sympy solve fallback).
    """

    def __init__(
        self,
        relations: Sequence[Relation],
        *,
        rel_tol: float = REL_TOL_DEFAULT,
        warn: WarnFunc = warnings.warn,
        lock_explicit: bool = False,
    ) -> None:
        """Initialize the solver state for a set of relations."""
        self.relations = relations
        self.rel_tol = rel_tol
        self.warn_sink = warn
        self._explicit: dict[str, sp.Expr] = {}
        self._explicit_tols: dict[str, float] = {}
        self._seeds: dict[str, sp.Expr] = {}
        self.lock_explicit = lock_explicit
        self._constraints_by_relation: dict[int, tuple[Relational, ...]] = {}
        for rel in self.relations:
            merged: list[Relational] = []
            if rel.constraints:
                for constraint in rel.constraints:
                    merged.append(parse_constraint(constraint))
            merged.extend(constraints_for_vars(rel.variables))
            if merged:
                self._constraints_by_relation[id(rel)] = tuple(merged)

    def set(
        self,
        name: str,
        value: sp.Expr | float | None,
        *,
        tol: float | None = None,
    ) -> None:
        """Record an explicit value for a variable with an optional tolerance."""
        expr = coerce_number(value, name)
        if expr is None:
            return
        self._explicit[name] = expr
        if tol is not None:
            self._explicit_tols[name] = float(tol)

    def seed(self, name: str, value: sp.Expr | float | None) -> None:
        """Add a default value that can be replaced by solved expressions."""
        if name in self._explicit:
            return
        expr = coerce_number(value, name)
        if expr is None:
            return
        self._seeds[name] = expr

    def solve(self, *, max_iterations: int = 50) -> dict[str, sp.Expr]:
        """Iteratively solve relations, honoring explicit values and constraints."""
        # Collect variable names from relations, explicit values, and seeds.
        names: list[str] = []
        seen: set[str] = set()
        for rel in self.relations:
            for var in rel.variables:
                if var in seen:
                    continue
                seen.add(var)
                names.append(var)
        for name in self._explicit:
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        for name in self._seeds:
            if name in seen:
                continue
            seen.add(name)
            names.append(name)

        symbols = {name: symbol(name) for name in names}
        values: dict[str, sp.Expr] = dict(self._explicit)
        explicit_names = set(self._explicit)
        seed_names = set()
        for name, value in self._seeds.items():
            if name in values:
                continue
            values[name] = value
            seed_names.add(name)

        def numeric_value(val: sp.Expr | float | int) -> float | None:
            """Return a numeric float if the value is fully resolved."""
            if isinstance(val, sp.Expr):
                if val.free_symbols:
                    return None
                evaluated = val.evalf(chop=True)
                if evaluated.is_real:
                    try:
                        return float(evaluated)
                    except (TypeError, ValueError):
                        return None
                try:
                    complex_val = complex(evaluated)
                except (TypeError, ValueError):
                    return None
                tol = 1e-12 * max(1.0, abs(complex_val.real))
                if abs(complex_val.imag) <= tol:
                    return float(complex_val.real)
                return None
            if isinstance(val, (int, float)):
                return float(val) if math.isfinite(val) else None
            return None

        def assign_value(name: str, new_value: sp.Expr) -> bool:
            """Update a variable if the new value is more resolved than the current."""
            numeric = numeric_value(new_value)
            if numeric is not None:
                new_value = sp.Float(numeric)
            existing = values.get(name)
            replace_seed = name in seed_names and new_value != existing
            should_replace = False
            if existing is None:
                should_replace = True
            elif isinstance(existing, sp.Expr) and isinstance(new_value, sp.Expr):
                should_replace = len(new_value.free_symbols) < len(existing.free_symbols)
            if should_replace or replace_seed:
                values[name] = new_value
                if name in seed_names:
                    seed_names.discard(name)
                return True
            return False

        def constraint_result(constraint: Relational, subs: dict[sp.Symbol, sp.Expr]) -> bool | None:
            """Return True/False for numeric constraints, or None if undecidable."""
            try:
                lhs_expr = constraint.lhs.subs(subs)
                rhs_expr = constraint.rhs.subs(subs)
            except Exception:
                lhs_expr = None
                rhs_expr = None
            lhs_val = numeric_value(lhs_expr) if lhs_expr is not None else None
            rhs_val = numeric_value(rhs_expr) if rhs_expr is not None else None
            if lhs_val is not None and rhs_val is not None:
                rel_op = getattr(constraint, "rel_op", None)
                op = _REL_OPS.get(rel_op)
                if op is None:
                    return None
                return op(lhs_val, rhs_val)
            try:
                evaluated = constraint.subs(subs)
            except Exception:
                return None
            if evaluated is True or evaluated == sp.true:
                return True
            if evaluated is False or evaluated == sp.false:
                return False
            if isinstance(evaluated, Relational):
                lhs_val = numeric_value(evaluated.lhs)
                rhs_val = numeric_value(evaluated.rhs)
                if lhs_val is None or rhs_val is None:
                    return None
                rel_op = getattr(evaluated, "rel_op", None)
                op = _REL_OPS.get(rel_op)
                if op is None:
                    return None
                return op(lhs_val, rhs_val)
            return None

        for _ in range(max_iterations):
            updated = False
            for rel in self.relations:
                if not rel.variables:
                    continue
                constraints = self._constraints_by_relation.get(id(rel), ())
                output_name = rel.variables[0]
                solve_targets = rel.solve_for or rel.variables
                var_values = {var: values.get(var, symbols[var]) for var in rel.variables}
                numeric_values = {var: numeric_value(val) for var, val in var_values.items()}
                output_numeric = numeric_values.get(output_name) is not None
                if not output_numeric:
                    if output_name not in solve_targets:
                        continue
                    missing_inputs = [var for var in rel.variables[1:] if numeric_values.get(var) is None]
                    if missing_inputs:
                        continue
                    output_sym = symbols[output_name]
                    output_expr = output_sym - rel.expr
                    subs = {symbols[var]: var_values[var] for var in rel.variables[1:]}
                    expr_sub = output_expr.subs(subs)
                    if output_sym in expr_sub.free_symbols:
                        continue
                    test_subs = dict(subs)
                    test_subs[output_sym] = expr_sub
                    if constraints:
                        constraint_ok = True
                        for constraint in constraints:
                            verdict = constraint_result(constraint, test_subs)
                            if verdict is False:
                                constraint_ok = False
                                break
                        if not constraint_ok:
                            continue
                    if assign_value(output_name, expr_sub):
                        updated = True
                    continue
                missing_inputs = [var for var in rel.variables[1:] if numeric_values.get(var) is None]
                if len(missing_inputs) != 1:
                    continue
                target = missing_inputs[0]
                if target not in solve_targets:
                    continue
                if target in explicit_names:
                    continue
                target_sym = symbols[target]
                exprs: list[sp.Expr] = []
                coeff = rel.expr.coeff(target_sym)
                if coeff is not None and coeff != 0:
                    rest = sp.simplify(rel.expr - coeff * target_sym)
                    if target_sym not in rest.free_symbols:
                        exprs = [-rest / coeff]
                if not exprs:
                    try:
                        exprs = sp.solve(rel.expr, target_sym)
                    except Exception:
                        exprs = []
                if isinstance(exprs, sp.Expr):
                    exprs = [exprs]
                if not exprs:
                    continue
                subs = {
                    symbols[var]: var_values[var]
                    for var in rel.variables
                    if var != target
                }
                # Choose a candidate expression for the target symbol.
                best: sp.Expr | None = None
                candidate: sp.Expr | None = None
                for expr in exprs:
                    expr_sub = expr.subs(subs)
                    if target_sym in expr_sub.free_symbols:
                        continue
                    test_subs = dict(subs)
                    test_subs[target_sym] = expr_sub
                    if constraints:
                        constraint_ok = True
                        for constraint in constraints:
                            verdict = constraint_result(constraint, test_subs)
                            if verdict is False:
                                constraint_ok = False
                                break
                        if not constraint_ok:
                            continue
                    if expr_sub.free_symbols:
                        if best is None:
                            best = expr_sub
                        continue
                    value = numeric_value(expr_sub)
                    if value is None:
                        if best is None:
                            best = expr_sub
                        continue
                    if value > 0:
                        candidate = expr_sub
                        break
                    if best is None:
                        best = expr_sub
                if candidate is None:
                    candidate = best
                if candidate is None:
                    continue
                if assign_value(target, candidate):
                    updated = True
            if not updated:
                break

        solution = {symbols[name]: values.get(name, symbols[name]) for name in names}

        # Emit warnings once per relation when explicit values violate tolerance.
        emitted_relations: set[str] = set()
        for rel in self.relations:
            if rel.name in emitted_relations:
                continue
            if not rel.variables:
                continue
            output_name = rel.variables[0]
            if output_name not in self._explicit:
                continue
            residual = rel.expr.subs(solution)
            residual_val = numeric_value(residual)
            if residual_val is None or not math.isfinite(residual_val):
                continue
            actual_expr = self._explicit[output_name]
            actual_sub = actual_expr.subs(solution) if isinstance(actual_expr, sp.Expr) else actual_expr
            actual_val = numeric_value(actual_sub)
            if actual_val is None:
                continue
            relation_val = actual_val - residual_val
            tol_scale = max(abs(relation_val), 1.0)
            rel_tol = self._explicit_tols.get(output_name, self.rel_tol)
            tolerance = rel_tol * tol_scale
            if abs(residual_val) <= tolerance:
                continue
            message = (
                f"{output_name} violates {rel.name} relation: expected value is {actual_val}, got {relation_val}"
            )
            context_vals: list[str] = []
            for name in rel.variables[1:]:
                sym = symbols.get(name)
                if sym is None:
                    continue
                value = solution.get(sym, sym)
                value_sub = value.subs(solution) if isinstance(value, sp.Expr) else value
                value_num = numeric_value(value_sub)
                if value_num is None:
                    continue
                context_vals.append(f"{name}={value_num}")
            if context_vals:
                message = f"{message} (holding {', '.join(context_vals)})"
            self.warn_sink(message, UserWarning)
            emitted_relations.add(rel.name)

        return {name: solution[symbols[name]] for name in names}
