from __future__ import annotations

import math
import operator
import warnings
from collections import deque
from typing import Callable, Mapping, Sequence

import networkx as nx
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
        rel_tol: float = REL_TOL_DEFAULT,
        solve_for: tuple[str, ...] | None = None,
        initial_guesses: Mapping[str, float | Callable[[Mapping[str, float]], float]] | None = None,
        constraints: Sequence[Relational | str] | None = None,
        backsolve_explicit: bool = False,
    ) -> None:
        """Store the metadata and symbolic form for a relation."""
        self.name = name
        self.variables = variables
        self.expr = sp.sympify(expr)
        self.rel_tol = rel_tol
        self.solve_for = solve_for
        self.initial_guesses = initial_guesses
        self.constraints = tuple(constraints or ())
        self.backsolve_explicit = bool(backsolve_explicit)


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
        self._rel_constraints_by_relation: dict[int, tuple[Relational, ...]] = {}
        self._var_constraints_by_relation: dict[int, tuple[Relational, ...]] = {}
        for rel in self.relations:
            rel_constraints: list[Relational] = []
            if rel.constraints:
                for constraint in rel.constraints:
                    rel_constraints.append(parse_constraint(constraint))
            var_constraints = list(constraints_for_vars(rel.variables))
            if rel_constraints:
                self._rel_constraints_by_relation[id(rel)] = tuple(rel_constraints)
            if var_constraints:
                self._var_constraints_by_relation[id(rel)] = tuple(var_constraints)
            merged = rel_constraints + var_constraints
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

    def _init_solve_state(self, *, global_mode: bool) -> None:
        """Initialize mutable solver state."""
        names: list[str] = []
        seen: set[str] = set()
        # does this take into account aliases?
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

        self._solve_names = names
        self._solve_symbols = {name: symbol(name) for name in names}
        values: dict[str, sp.Expr] = dict(self._explicit)
        explicit_names = set(self._explicit)
        seed_names: set[str] = set()
        for name, value in self._seeds.items():
            if name in values:
                continue
            values[name] = value
            seed_names.add(name)
        self._solve_values = values
        self._solve_explicit_names = explicit_names
        self._solve_seed_names = seed_names

        allowed_unknowns: set[str] = set()
        for rel in self.relations:
            targets = rel.solve_for or rel.variables
            allowed_unknowns.update(targets)
        if self.lock_explicit:
            allowed_unknowns.difference_update(explicit_names)
        self._solve_allowed_unknowns = allowed_unknowns

        self._solve_guess_cache = {}
        for name, value in values.items():
            num = self._numeric_value(value)
            if num is not None:
                self._solve_guess_cache[name] = float(num)

        rels_by_var: dict[str, list[Relation]] = {}
        for rel in self.relations:
            if not rel.variables:
                continue
            for var in rel.variables:
                rels_by_var.setdefault(var, []).append(rel)
        self._solve_rels_by_var = rels_by_var
        self._solve_queue = deque(rel for rel in self.relations if rel.variables)
        self._solve_queued = set(self._solve_queue)

        if global_mode:
            self._solve_coupled_unknowns = 6
            self._solve_coupled_relations = 8
            self._solve_coupled_nsolve = True
        else:
            self._solve_coupled_unknowns = 2
            self._solve_coupled_relations = 3
            self._solve_coupled_nsolve = False
        self._solve_global_mode = global_mode
        self._solve_allow_explicit_output = True

    def _numeric_value(self, val: sp.Expr | float | int, *, _seen: tuple[str, ...] = ()) -> float | None:
        """Return a numeric float if the value is fully resolved."""
        if isinstance(val, sp.Expr):
            if val.free_symbols:
                subs: dict[sp.Symbol, sp.Expr] = {}
                for sym in val.free_symbols:
                    name = sym.name
                    if name in _seen:
                        return None
                    ref = self._solve_values.get(name)
                    if ref is None:
                        return None
                    ref_num = self._numeric_value(ref, _seen=_seen + (name,))
                    if ref_num is None:
                        return None
                    subs[sym] = sp.Float(ref_num)
                try:
                    val = val.subs(subs)
                except Exception:
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

    def _assign_value(
        self,
        name: str,
        new_value: sp.Expr,
        *,
        replace_numeric: bool = False,
        rel_tol: float | None = None,
    ) -> bool:
        """Update a variable if the new value is more resolved than the current."""
        numeric = self._numeric_value(new_value)
        if numeric is not None:
            self._solve_guess_cache[name] = float(numeric)
            new_value = sp.Float(numeric)
        existing = self._solve_values.get(name)
        replace_seed = name in self._solve_seed_names and new_value != existing
        should_replace = False
        if existing is None:
            should_replace = True
        elif isinstance(existing, sp.Expr) and isinstance(new_value, sp.Expr):
            existing_symbols = existing.free_symbols
            new_symbols = new_value.free_symbols
            if len(new_symbols) < len(existing_symbols):
                should_replace = True
            elif replace_numeric and not existing_symbols and not new_symbols:
                existing_num = self._numeric_value(existing)
                new_num = self._numeric_value(new_value)
                if existing_num is not None and new_num is not None:
                    tol = self.rel_tol if rel_tol is None else rel_tol
                    tol_scale = max(abs(existing_num), abs(new_num), 1.0)
                    if abs(existing_num - new_num) > tol * tol_scale:
                        should_replace = True
        if should_replace or replace_seed:
            self._solve_values[name] = new_value
            if name in self._solve_seed_names:
                self._solve_seed_names.discard(name)
            return True
        return False

    def _constraint_result(self, constraint: Relational, subs: dict[sp.Symbol, sp.Expr]) -> bool | None:
        """Return True/False for numeric constraints, or None if undecidable."""
        try:
            lhs_expr = constraint.lhs.subs(subs)
            rhs_expr = constraint.rhs.subs(subs)
        except Exception:
            lhs_expr = None
            rhs_expr = None
        lhs_val = self._numeric_value(lhs_expr) if lhs_expr is not None else None
        rhs_val = self._numeric_value(rhs_expr) if rhs_expr is not None else None
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
            lhs_val = self._numeric_value(evaluated.lhs)
            rhs_val = self._numeric_value(evaluated.rhs)
            if lhs_val is None or rhs_val is None:
                return None
            rel_op = getattr(evaluated, "rel_op", None)
            op = _REL_OPS.get(rel_op)
            if op is None:
                return None
            return op(lhs_val, rhs_val)
        return None

    def _constraints_for_candidate(
        self,
        rel: Relation,
        unknown_syms: set[sp.Symbol],
    ) -> tuple[Relational, ...]:
        rel_constraints = self._rel_constraints_by_relation.get(id(rel), ())
        var_constraints = self._var_constraints_by_relation.get(id(rel), ())
        if not var_constraints or not unknown_syms:
            return rel_constraints
        filtered = list(rel_constraints)
        for constraint in var_constraints:
            if constraint.free_symbols & unknown_syms:
                filtered.append(constraint)
        return tuple(filtered)

    def _build_guess_values(
        self,
        component_rels: Sequence[Relation],
        unknown_syms: Sequence[sp.Symbol],
    ) -> dict[str, float]:
        guess_values: dict[str, float] = {}
        unknown_names = {sym.name for sym in unknown_syms}
        for sym in unknown_syms:
            num = self._numeric_value(self._solve_values.get(sym.name, sym))
            if num is not None:
                guess_values[sym.name] = float(num)
        for sym in unknown_syms:
            if sym.name in guess_values:
                continue
            cached = self._solve_guess_cache.get(sym.name)
            if cached is not None and math.isfinite(cached):
                guess_values[sym.name] = float(cached)

        def context() -> dict[str, float]:
            ctx: dict[str, float] = {}
            for key, val in self._solve_values.items():
                num = self._numeric_value(val)
                if num is not None:
                    ctx[key] = float(num)
            ctx.update(guess_values)
            return ctx

        for rel_guess in component_rels:
            if not rel_guess.initial_guesses:
                continue
            ctx = context()
            for name, guesser in rel_guess.initial_guesses.items():
                if name in guess_values:
                    continue
                try:
                    guess = guesser(ctx) if callable(guesser) else guesser
                except Exception:
                    continue
                if isinstance(guess, (int, float)) and math.isfinite(float(guess)):
                    guess_values[name] = float(guess)

        progress = True
        while progress:
            progress = False
            ctx = context()
            for rel_guess in component_rels:
                for target in rel_guess.variables:
                    if target not in unknown_names or target in guess_values:
                        continue
                    if any(name not in ctx for name in rel_guess.variables if name != target):
                        continue
                    target_sym = self._solve_symbols[target]
                    coeff = rel_guess.expr.coeff(target_sym)
                    if coeff is None or coeff == 0:
                        continue
                    rest = sp.simplify(rel_guess.expr - coeff * target_sym)
                    if target_sym in rest.free_symbols:
                        continue
                    subs = {
                        self._solve_symbols[name]: sp.Float(ctx[name])
                        for name in rel_guess.variables
                        if name != target
                    }
                    expr_sub = (-rest / coeff).subs(subs)
                    num = self._numeric_value(expr_sub)
                    if num is None or not math.isfinite(float(num)):
                        continue
                    guess_values[target] = float(num)
                    progress = True
        reverse_rel_op = {
            "<": ">",
            ">": "<",
            "<=": ">=",
            ">=": "<=",
            "==": "==",
            "!=": "!=",
        }

        def guess_from_constraints(name: str) -> float | None:
            sym = self._solve_symbols.get(name)
            if sym is None:
                return None
            for rel_guess in component_rels:
                constraints = self._constraints_by_relation.get(id(rel_guess), ())
                for constraint in constraints:
                    if not isinstance(constraint, Relational):
                        continue
                    rel_op = getattr(constraint, "rel_op", None)
                    if rel_op not in _REL_OPS:
                        continue
                    lhs = constraint.lhs
                    rhs = constraint.rhs
                    if lhs == sym and not rhs.free_symbols:
                        rhs_val = self._numeric_value(rhs)
                        if rhs_val is None:
                            continue
                        if rel_op == "==":
                            return float(rhs_val)
                        if rel_op in (">", ">="):
                            return float(rhs_val) + 1.0
                        if rel_op in ("<", "<="):
                            return float(rhs_val) - 1.0
                        if rel_op == "!=":
                            return float(rhs_val) + 1.0
                    if rhs == sym and not lhs.free_symbols:
                        lhs_val = self._numeric_value(lhs)
                        if lhs_val is None:
                            continue
                        rel_op_rev = reverse_rel_op.get(rel_op)
                        if rel_op_rev is None:
                            continue
                        if rel_op_rev == "==":
                            return float(lhs_val)
                        if rel_op_rev in (">", ">="):
                            return float(lhs_val) + 1.0
                        if rel_op_rev in ("<", "<="):
                            return float(lhs_val) - 1.0
                        if rel_op_rev == "!=":
                            return float(lhs_val) + 1.0
            return None

        for sym in unknown_syms:
            if sym.name in guess_values:
                continue
            guess = guess_from_constraints(sym.name)
            if guess is not None and math.isfinite(guess):
                guess_values[sym.name] = float(guess)

        for sym in unknown_syms:
            if sym.name in guess_values:
                continue
            guess_values[sym.name] = 1.0
        return guess_values

    def _attempt_solve(
        self,
        component_rels: Sequence[Relation],
        component_vars: set[str],
        *,
        allow_nsolve: bool,
    ) -> set[str]:
        updated_vars: set[str] = set()
        if not component_rels or not component_vars:
            return updated_vars
        unknown_syms = [self._solve_symbols[var] for var in sorted(component_vars)]
        if not unknown_syms:
            return updated_vars

        subs_known: dict[sp.Symbol, sp.Expr] = {}
        for rel_in_comp in component_rels:
            for var in rel_in_comp.variables:
                val = self._solve_values.get(var, self._solve_symbols[var])
                num = self._numeric_value(val)
                if num is not None:
                    subs_known[self._solve_symbols[var]] = sp.Float(num)

        equations: list[sp.Expr] = []
        allowed = set(unknown_syms)
        for rel_in_comp in component_rels:
            expr = rel_in_comp.expr.subs(subs_known)
            if expr == 0:
                continue
            if expr.free_symbols - allowed:
                equations = []
                break
            equations.append(expr)
        if not equations or len(equations) < len(unknown_syms):
            return updated_vars

        solutions: list[dict[sp.Symbol, sp.Expr]] = []
        is_linear = False

        try:
            matrix, rhs = sp.linear_eq_to_matrix(equations, unknown_syms)
        except Exception:
            matrix = None
            rhs = None
        if matrix is not None and rhs is not None:
            is_linear = True
            try:
                linear_solutions = sp.linsolve((matrix, rhs), unknown_syms)
            except Exception:
                linear_solutions = None
            if linear_solutions:
                for sol in linear_solutions:
                    solutions.append(dict(zip(unknown_syms, sol)))

        polynomial = True
        if not solutions and not is_linear:
            for expr in equations:
                try:
                    if not expr.is_polynomial(*unknown_syms):
                        polynomial = False
                        break
                except Exception:
                    polynomial = False
                    break
        if not solutions and polynomial:
            try:
                solve_output = sp.solve(equations, unknown_syms, dict=True)
            except Exception:
                solve_output = []
            if isinstance(solve_output, dict):
                solve_output = [solve_output]
            if solve_output:
                solutions.extend(solve_output)

        unknown_set = set(unknown_syms)

        def candidate_from_solution(
            sol: dict[sp.Symbol, sp.Expr],
        ) -> dict[sp.Symbol, sp.Expr] | None:
            if any(sym not in sol for sym in unknown_syms):
                return None
            candidate: dict[sp.Symbol, sp.Expr] = {}
            for sym in unknown_syms:
                expr_sub = sol[sym].subs(subs_known)
                if expr_sub.free_symbols:
                    return None
                num = self._numeric_value(expr_sub)
                if num is None:
                    return None
                candidate[sym] = sp.Float(num)

            for rel_in_comp in component_rels:
                constraints = self._constraints_for_candidate(rel_in_comp, unknown_set)
                if not constraints:
                    continue
                test_subs: dict[sp.Symbol, sp.Expr] = {}
                for var in rel_in_comp.variables:
                    sym = self._solve_symbols[var]
                    if sym in candidate:
                        test_subs[sym] = candidate[sym]
                        continue
                    val = self._solve_values.get(var, sym)
                    num = self._numeric_value(val)
                    if num is None:
                        return None
                    test_subs[sym] = sp.Float(num)
                for constraint in constraints:
                    verdict = self._constraint_result(constraint, test_subs)
                    if verdict is False:
                        return None
            return candidate

        candidate_solution: dict[sp.Symbol, sp.Expr] | None = None
        nsolve_candidate = False
        for sol in solutions:
            candidate_solution = candidate_from_solution(sol)
            if candidate_solution is not None:
                break

        if (
            candidate_solution is None
            and allow_nsolve
            and not is_linear
            and len(equations) == len(unknown_syms)
        ):
            guess_values = self._build_guess_values(component_rels, unknown_syms)
            if len(guess_values) == len(unknown_syms):
                guesses = [guess_values[sym.name] for sym in unknown_syms]
                try:
                    nsolve_solution = sp.nsolve(equations, unknown_syms, guesses, tol=1e-10, maxsteps=50)
                except Exception:
                    try:
                        nsolve_solution = sp.nsolve(equations, unknown_syms, guesses, maxsteps=50)
                    except Exception:
                        nsolve_solution = None
                if nsolve_solution is not None:
                    if len(unknown_syms) == 1:
                        sol_values = [nsolve_solution]
                    else:
                        try:
                            sol_values = list(nsolve_solution)
                        except TypeError:
                            sol_values = [nsolve_solution]
                    if len(sol_values) == len(unknown_syms):
                        candidate_solution = candidate_from_solution(dict(zip(unknown_syms, sol_values)))
                        if candidate_solution is not None:
                            nsolve_candidate = True

        if candidate_solution is None:
            return updated_vars
        if nsolve_candidate:
            try:
                jacobian = sp.Matrix(equations).jacobian(unknown_syms)
                subs = {sym: candidate_solution[sym] for sym in unknown_syms}
                jacobian = jacobian.subs(subs)
                if jacobian.rank() < len(unknown_syms):
                    return updated_vars
            except Exception:
                pass

        for sym in unknown_syms:
            if self._assign_value(sym.name, candidate_solution[sym]):
                updated_vars.add(sym.name)
        return updated_vars

    def _process_relation(self, rel: Relation) -> set[str]:
        updated_vars: set[str] = set()
        if not rel.variables:
            return updated_vars
        output_sym = self._solve_symbols[rel.variables[0]]
        constraints_for_output = self._constraints_for_candidate(rel, {output_sym})
        output_name = rel.variables[0]
        solve_targets = rel.solve_for or rel.variables
        var_values = {var: self._solve_values.get(var, self._solve_symbols[var]) for var in rel.variables}
        numeric_values = {var: self._numeric_value(val) for var, val in var_values.items()}
        output_numeric = numeric_values.get(output_name) is not None
        if not output_numeric:
            if (
                self.lock_explicit
                and not self._solve_allow_explicit_output
                and output_name in self._solve_explicit_names
                and not rel.backsolve_explicit
            ):
                return updated_vars
            if output_name not in solve_targets:
                return updated_vars
            missing_inputs = [var for var in rel.variables[1:] if numeric_values.get(var) is None]
            if missing_inputs:
                return updated_vars
            output_expr = output_sym - rel.expr
            subs = {self._solve_symbols[var]: var_values[var] for var in rel.variables[1:]}
            expr_sub = output_expr.subs(subs)
            if output_sym in expr_sub.free_symbols:
                return updated_vars
            test_subs = dict(subs)
            test_subs[output_sym] = expr_sub
            if constraints_for_output:
                constraint_ok = True
                for constraint in constraints_for_output:
                    verdict = self._constraint_result(constraint, test_subs)
                    if verdict is False:
                        constraint_ok = False
                        break
                if not constraint_ok:
                    return updated_vars
            if self._assign_value(output_name, expr_sub):
                updated_vars.add(output_name)
            return updated_vars

        if (
            self.lock_explicit
            and not self._solve_allow_explicit_output
            and output_name in self._solve_explicit_names
            and not rel.backsolve_explicit
        ):
            return updated_vars

        inputs = rel.variables[1:]
        inputs_numeric = all(numeric_values.get(var) is not None for var in inputs)
        if (
            inputs_numeric
            and output_name in solve_targets
            and output_name not in self._solve_explicit_names
        ):
            rel_tol = rel.rel_tol
            subs_all = {
                self._solve_symbols[var]: sp.Float(numeric_values[var])
                for var in rel.variables
                if numeric_values.get(var) is not None
            }
            residual = rel.expr.subs(subs_all)
            residual_val = self._numeric_value(residual)
            if residual_val is not None:
                output_val = numeric_values.get(output_name)
                tol_scale = max(abs(output_val), 1.0) if output_val is not None else 1.0
                if abs(residual_val) > rel_tol * tol_scale:
                    output_expr = output_sym - rel.expr
                    subs = {
                        self._solve_symbols[var]: sp.Float(numeric_values[var])
                        for var in inputs
                    }
                    expr_sub = output_expr.subs(subs)
                    if output_sym in expr_sub.free_symbols:
                        return updated_vars
                    test_subs = dict(subs)
                    test_subs[output_sym] = expr_sub
                    if constraints_for_output:
                        constraint_ok = True
                        for constraint in constraints_for_output:
                            verdict = self._constraint_result(constraint, test_subs)
                            if verdict is False:
                                constraint_ok = False
                                break
                        if not constraint_ok:
                            return updated_vars
                    if self._assign_value(
                        output_name,
                        expr_sub,
                        replace_numeric=True,
                        rel_tol=rel_tol,
                    ):
                        updated_vars.add(output_name)
                    return updated_vars

        missing_inputs = [var for var in inputs if numeric_values.get(var) is None]
        if len(missing_inputs) != 1:
            return updated_vars
        target = missing_inputs[0]
        if target not in solve_targets:
            return updated_vars
        if target in self._solve_explicit_names:
            return updated_vars
        target_sym = self._solve_symbols[target]
        constraints_for_target = self._constraints_for_candidate(rel, {target_sym})
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
            return updated_vars
        subs = {
            self._solve_symbols[var]: var_values[var]
            for var in rel.variables
            if var != target
        }
        best: sp.Expr | None = None
        candidate: sp.Expr | None = None
        for expr in exprs:
            expr_sub = expr.subs(subs)
            if target_sym in expr_sub.free_symbols:
                continue
            test_subs = dict(subs)
            test_subs[target_sym] = expr_sub
            if constraints_for_target:
                constraint_ok = True
                for constraint in constraints_for_target:
                    verdict = self._constraint_result(constraint, test_subs)
                    if verdict is False:
                        constraint_ok = False
                        break
                if not constraint_ok:
                    continue
            if expr_sub.free_symbols:
                if best is None:
                    best = expr_sub
                continue
            value = self._numeric_value(expr_sub)
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
            return updated_vars
        if self._assign_value(target, candidate):
            updated_vars.add(target)
        return updated_vars

    def _emit_relation_warnings(self, solution: dict[sp.Symbol, sp.Expr]) -> None:
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
            residual_val = self._numeric_value(residual)
            if residual_val is None or not math.isfinite(residual_val):
                continue
            actual_expr = self._explicit[output_name]
            actual_sub = actual_expr.subs(solution) if isinstance(actual_expr, sp.Expr) else actual_expr
            actual_val = self._numeric_value(actual_sub)
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
                sym = self._solve_symbols.get(name)
                if sym is None:
                    continue
                value = solution.get(sym, sym)
                value_sub = value.subs(solution) if isinstance(value, sp.Expr) else value
                value_num = self._numeric_value(value_sub)
                if value_num is None:
                    continue
                context_vals.append(f"{name}={value_num}")
            if context_vals:
                message = f"{message} (holding {', '.join(context_vals)})"
            self.warn_sink(message, UserWarning)
            emitted_relations.add(rel.name)

    def solve(self, *, max_iterations: int = 50, global_mode: bool = False) -> dict[str, sp.Expr]:
        """Iteratively solve relations, honoring explicit values and constraints."""
        self._init_solve_state(global_mode=global_mode)
        phase_modes = (True,) if not self.lock_explicit else (False, True)
        for allow_explicit in phase_modes:
            # Step 1: configure explicit-output policy and initialize the work queue.
            self._solve_allow_explicit_output = allow_explicit
            self._solve_queue = deque(rel for rel in self.relations if rel.variables)
            self._solve_queued = set(self._solve_queue)
            for _ in range(max_iterations):
                # Step 2: walk single-unknown relations until no progress is made.
                updated = False
                while self._solve_queue:
                    rel = self._solve_queue.popleft()
                    self._solve_queued.discard(rel)
                    updated_vars = self._process_relation(rel)
                    if not updated_vars:
                        continue
                    updated = True
                    for var in updated_vars:
                        for rel_dep in self._solve_rels_by_var.get(var, ()):
                            if rel_dep in self._solve_queued:
                                continue
                            self._solve_queue.append(rel_dep)
                            self._solve_queued.add(rel_dep)
                if updated:
                    continue

                # Step 3: compute unknowns for each relation and index by variable.
                unknowns_by_relation: dict[Relation, tuple[str, ...]] = {}
                rels_by_var_unknowns: dict[str, list[tuple[Relation, tuple[str, ...]]]] = {}
                for rel in self.relations:
                    if not rel.variables:
                        continue
                    if (
                        self.lock_explicit
                        and not self._solve_allow_explicit_output
                        and rel.variables[0] in self._solve_explicit_names
                        and not rel.backsolve_explicit
                    ):
                        continue
                    solve_targets = rel.solve_for or rel.variables
                    unknowns: list[str] = []
                    for var in rel.variables:
                        val = self._solve_values.get(var, self._solve_symbols[var])
                        if self._numeric_value(val) is None:
                            if var not in solve_targets:
                                unknowns = []
                                break
                            if var not in self._solve_allowed_unknowns:
                                unknowns = []
                                break
                            unknowns.append(var)
                    if not unknowns:
                        continue
                    unknowns_tuple = tuple(unknowns)
                    unknowns_by_relation[rel] = unknowns_tuple
                    for var in unknowns_tuple:
                        rels_by_var_unknowns.setdefault(var, []).append((rel, unknowns_tuple))
                if not unknowns_by_relation:
                    break

                # Step 4: build a bipartite graph of relations versus unknown variables.
                graph = nx.Graph()
                for rel, unknowns in unknowns_by_relation.items():
                    graph.add_node(rel, bipartite=0)
                    for var in unknowns:
                        graph.add_node(var, bipartite=1)
                        graph.add_edge(rel, var)

                # Step 5: peel degree-1 variable nodes to isolate the coupled core.
                queue = deque(
                    node
                    for node, data in graph.nodes(data=True)
                    if data.get("bipartite") == 1 and graph.degree(node) == 1
                )
                while queue:
                    var = queue.popleft()
                    if var not in graph:
                        continue
                    if graph.degree(var) != 1:
                        continue
                    neighbors = list(graph.neighbors(var))
                    if not neighbors:
                        graph.remove_node(var)
                        continue
                    rel = neighbors[0]
                    other_vars: list[str] = []
                    if rel in graph:
                        other_vars = [nbr for nbr in graph.neighbors(rel) if nbr != var]
                    graph.remove_node(var)
                    if rel in graph:
                        graph.remove_node(rel)
                    for other_var in other_vars:
                        if other_var in graph and graph.nodes[other_var].get("bipartite") == 1:
                            if graph.degree(other_var) == 1:
                                queue.append(other_var)

                # Step 6: compute coupled blocks using matching and SCC ordering.
                blocks: list[tuple[list[Relation], set[str]]] = []
                if graph.number_of_edges() > 0:
                    rel_nodes = {n for n, data in graph.nodes(data=True) if data.get("bipartite") == 0}
                    var_nodes = {n for n, data in graph.nodes(data=True) if data.get("bipartite") == 1}
                    if rel_nodes and var_nodes:
                        matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=rel_nodes)
                        directed = nx.DiGraph()
                        directed.add_nodes_from(graph.nodes())
                        for rel in rel_nodes:
                            for var in graph.neighbors(rel):
                                if matching.get(rel) == var:
                                    directed.add_edge(var, rel)
                                else:
                                    directed.add_edge(rel, var)
                        if directed.number_of_edges() > 0:
                            sccs = list(nx.strongly_connected_components(directed))
                            condensed = nx.condensation(directed, sccs)
                            ordered_components = [
                                condensed.nodes[idx]["members"]
                                for idx in nx.topological_sort(condensed)
                            ]
                            for component in ordered_components:
                                component_rels = [node for node in component if node in rel_nodes]
                                component_vars = {node for node in component if node in var_nodes}
                                if not component_rels or not component_vars:
                                    continue
                                blocks.append((component_rels, component_vars))

                # Step 7: attempt coupled solves in increasing size and relation count.
                coupled_updates: set[str] = set()
                for size in range(2, self._solve_coupled_unknowns + 1):
                    for rel_count in range(size, self._solve_coupled_relations + 1):
                        for component_rels, component_vars in blocks:
                            if len(component_vars) > size or len(component_rels) > rel_count:
                                continue
                            coupled_updates.update(
                                self._attempt_solve(
                                    component_rels,
                                    component_vars,
                                    allow_nsolve=self._solve_coupled_nsolve,
                                )
                            )
                        if coupled_updates:
                            break
                        if global_mode:
                            for target in sorted(self._solve_allowed_unknowns):
                                if self.lock_explicit and target in self._solve_explicit_names:
                                    continue
                                val = self._solve_values.get(target, self._solve_symbols[target])
                                if self._numeric_value(val) is not None:
                                    continue
                                candidates: list[tuple[int, int, Relation, tuple[str, ...]]] = []
                                for rel, unknowns in rels_by_var_unknowns.get(target, ()):
                                    candidates.append((len(unknowns), len(rel.variables), rel, unknowns))
                                if not candidates:
                                    continue
                                candidates.sort(key=lambda item: (item[0], item[1]))
                                if len(candidates) > rel_count:
                                    candidates = candidates[:rel_count]
                                component_rels = []
                                component_vars: set[str] = set()
                                for _, _, rel, unknowns in candidates:
                                    component_rels.append(rel)
                                    component_vars.update(unknowns)
                                if not component_vars or len(component_vars) > size:
                                    continue
                                if len(component_rels) < len(component_vars):
                                    continue
                                coupled_updates.update(
                                    self._attempt_solve(
                                        component_rels,
                                        component_vars,
                                        allow_nsolve=self._solve_coupled_nsolve,
                                    )
                                )
                                if coupled_updates:
                                    break
                            if coupled_updates:
                                break
                    if coupled_updates:
                        break
                if not coupled_updates:
                    break

                # Step 8: enqueue updates for another single-unknown pass.
                for var in coupled_updates:
                    for rel_dep in self._solve_rels_by_var.get(var, ()):
                        if rel_dep in self._solve_queued:
                            continue
                        self._solve_queue.append(rel_dep)
                        self._solve_queued.add(rel_dep)

        solution = {
            self._solve_symbols[name]: self._solve_values.get(name, self._solve_symbols[name])
            for name in self._solve_names
        }
        self._emit_relation_warnings(solution)
        return {name: solution[self._solve_symbols[name]] for name in self._solve_names}
