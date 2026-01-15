from __future__ import annotations

import math
import operator
import warnings
from collections import deque
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

    def solve(self, *, max_iterations: int = 50, global_mode: bool = False) -> dict[str, sp.Expr]:
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

        def numeric_value(val: sp.Expr | float | int, *, _seen: tuple[str, ...] = ()) -> float | None:
            """Return a numeric float if the value is fully resolved."""
            if isinstance(val, sp.Expr):
                if val.free_symbols:
                    subs: dict[sp.Symbol, sp.Expr] = {}
                    for sym in val.free_symbols:
                        name = sym.name
                        if name in _seen:
                            return None
                        ref = values.get(name)
                        if ref is None:
                            return None
                        ref_num = numeric_value(ref, _seen=_seen + (name,))
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

        guess_cache: dict[str, float] = {}

        def assign_value(name: str, new_value: sp.Expr) -> bool:
            """Update a variable if the new value is more resolved than the current."""
            numeric = numeric_value(new_value)
            if numeric is not None:
                guess_cache[name] = float(numeric)
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

        for name, value in values.items():
            num = numeric_value(value)
            if num is not None:
                guess_cache[name] = float(num)

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

        allowed_unknowns: set[str] = set()
        for rel in self.relations:
            targets = rel.solve_for or rel.variables
            allowed_unknowns.update(targets)
        if self.lock_explicit:
            allowed_unknowns.difference_update(explicit_names)

        def relation_unknowns(
            rel: Relation,
            *,
            max_unknowns: int | None = None,
        ) -> tuple[str, ...] | None:
            if not rel.variables:
                return None
            solve_targets = rel.solve_for or rel.variables
            unknowns: list[str] = []
            for var in rel.variables:
                val = values.get(var, symbols[var])
                if numeric_value(val) is None:
                    if var not in solve_targets:
                        return None
                    if var not in allowed_unknowns:
                        return None
                    unknowns.append(var)
            if not unknowns:
                return None
            if max_unknowns is not None and len(unknowns) > max_unknowns:
                return None
            return tuple(unknowns)

        def build_guess_values(
            component_rels: Sequence[Relation],
            unknown_syms: Sequence[sp.Symbol],
        ) -> dict[str, float]:
            guess_values: dict[str, float] = {}
            unknown_names = {sym.name for sym in unknown_syms}
            for sym in unknown_syms:
                num = numeric_value(values.get(sym.name, sym))
                if num is not None:
                    guess_values[sym.name] = float(num)
            for sym in unknown_syms:
                if sym.name in guess_values:
                    continue
                cached = guess_cache.get(sym.name)
                if cached is not None and math.isfinite(cached):
                    guess_values[sym.name] = float(cached)

            def context() -> dict[str, float]:
                ctx: dict[str, float] = {}
                for key, val in values.items():
                    num = numeric_value(val)
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
                        target_sym = symbols[target]
                        coeff = rel_guess.expr.coeff(target_sym)
                        if coeff is None or coeff == 0:
                            continue
                        rest = sp.simplify(rel_guess.expr - coeff * target_sym)
                        if target_sym in rest.free_symbols:
                            continue
                        subs = {
                            symbols[name]: sp.Float(ctx[name])
                            for name in rel_guess.variables
                            if name != target
                        }
                        expr_sub = (-rest / coeff).subs(subs)
                        num = numeric_value(expr_sub)
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
                sym = symbols.get(name)
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
                            rhs_val = numeric_value(rhs)
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
                            lhs_val = numeric_value(lhs)
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

        def attempt_solve(
            component_rels: Sequence[Relation],
            component_vars: set[str],
            *,
            allow_nsolve: bool,
        ) -> set[str]:
            updated_vars: set[str] = set()
            if not component_rels or not component_vars:
                return updated_vars
            unknown_syms = [symbols[var] for var in sorted(component_vars)]
            if not unknown_syms:
                return updated_vars

            subs_known: dict[sp.Symbol, sp.Expr] = {}
            for rel_in_comp in component_rels:
                for var in rel_in_comp.variables:
                    val = values.get(var, symbols[var])
                    num = numeric_value(val)
                    if num is not None:
                        subs_known[symbols[var]] = sp.Float(num)

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

            try:
                matrix, rhs = sp.linear_eq_to_matrix(equations, unknown_syms)
            except Exception:
                matrix = None
                rhs = None
            if matrix is not None and rhs is not None:
                try:
                    linear_solutions = sp.linsolve((matrix, rhs), unknown_syms)
                except Exception:
                    linear_solutions = None
                if linear_solutions:
                    for sol in linear_solutions:
                        solutions.append(dict(zip(unknown_syms, sol)))

            polynomial = True
            if not solutions:
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

            def candidate_from_solution(sol: dict[sp.Symbol, sp.Expr]) -> dict[sp.Symbol, sp.Expr] | None:
                if any(sym not in sol for sym in unknown_syms):
                    return None
                candidate: dict[sp.Symbol, sp.Expr] = {}
                for sym in unknown_syms:
                    expr_sub = sol[sym].subs(subs_known)
                    if expr_sub.free_symbols:
                        return None
                    num = numeric_value(expr_sub)
                    if num is None:
                        return None
                    candidate[sym] = sp.Float(num)

                for rel_in_comp in component_rels:
                    constraints = self._constraints_by_relation.get(id(rel_in_comp), ())
                    if not constraints:
                        continue
                    test_subs: dict[sp.Symbol, sp.Expr] = {}
                    for var in rel_in_comp.variables:
                        sym = symbols[var]
                        if sym in candidate:
                            test_subs[sym] = candidate[sym]
                            continue
                        val = values.get(var, sym)
                        num = numeric_value(val)
                        if num is None:
                            return None
                        test_subs[sym] = sp.Float(num)
                    for constraint in constraints:
                        verdict = constraint_result(constraint, test_subs)
                        if verdict is False:
                            return None
                return candidate

            candidate_solution: dict[sp.Symbol, sp.Expr] | None = None
            for sol in solutions:
                candidate_solution = candidate_from_solution(sol)
                if candidate_solution is not None:
                    break

            if candidate_solution is None and allow_nsolve and len(equations) == len(unknown_syms):
                guess_values = build_guess_values(component_rels, unknown_syms)
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

            if candidate_solution is None:
                return updated_vars

            for sym in unknown_syms:
                if assign_value(sym.name, candidate_solution[sym]):
                    updated_vars.add(sym.name)
            return updated_vars

        def process_relation(rel: Relation) -> set[str]:
            updated_vars: set[str] = set()
            if not rel.variables:
                return updated_vars
            constraints = self._constraints_by_relation.get(id(rel), ())
            output_name = rel.variables[0]
            solve_targets = rel.solve_for or rel.variables
            var_values = {var: values.get(var, symbols[var]) for var in rel.variables}
            numeric_values = {var: numeric_value(val) for var, val in var_values.items()}
            output_numeric = numeric_values.get(output_name) is not None
            if not output_numeric:
                if self.lock_explicit and output_name in explicit_names:
                    return updated_vars
                if output_name not in solve_targets:
                    return updated_vars
                missing_inputs = [var for var in rel.variables[1:] if numeric_values.get(var) is None]
                if missing_inputs:
                    return updated_vars
                output_sym = symbols[output_name]
                output_expr = output_sym - rel.expr
                subs = {symbols[var]: var_values[var] for var in rel.variables[1:]}
                expr_sub = output_expr.subs(subs)
                if output_sym in expr_sub.free_symbols:
                    return updated_vars
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
                        return updated_vars
                if assign_value(output_name, expr_sub):
                    updated_vars.add(output_name)
                return updated_vars

            missing_inputs = [var for var in rel.variables[1:] if numeric_values.get(var) is None]
            if len(missing_inputs) != 1:
                return updated_vars
            target = missing_inputs[0]
            if target not in solve_targets:
                return updated_vars
            if target in explicit_names:
                return updated_vars
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
                return updated_vars
            subs = {
                symbols[var]: var_values[var]
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
                return updated_vars
            if assign_value(target, candidate):
                updated_vars.add(target)
            return updated_vars

        def build_relation_graph() -> tuple[dict[Relation, set[str]], dict[str, set[Relation]]]:
            rel_to_vars: dict[Relation, set[str]] = {}
            var_to_rels: dict[str, set[Relation]] = {}
            for rel in self.relations:
                unknowns = relation_unknowns(rel)
                if unknowns is None:
                    continue
                rel_to_vars[rel] = set(unknowns)
                for var in unknowns:
                    var_to_rels.setdefault(var, set()).add(rel)
            return rel_to_vars, var_to_rels

        def peel_leaf_pairs(
            rel_to_vars: dict[Relation, set[str]],
            var_to_rels: dict[str, set[Relation]],
        ) -> tuple[set[Relation], set[str]]:
            active_rels = set(rel_to_vars)
            active_vars = set(var_to_rels)
            queue = deque([var for var, rels in var_to_rels.items() if len(rels) == 1])
            while queue:
                var = queue.popleft()
                if var not in active_vars:
                    continue
                rels = var_to_rels.get(var)
                if not rels or len(rels) != 1:
                    active_vars.discard(var)
                    continue
                rel = next(iter(rels))
                if rel not in active_rels:
                    active_vars.discard(var)
                    continue
                active_vars.discard(var)
                active_rels.discard(rel)
                for other_var in rel_to_vars.get(rel, ()):
                    var_to_rels[other_var].discard(rel)
                    if other_var in active_vars and len(var_to_rels[other_var]) == 1:
                        queue.append(other_var)
                rel_to_vars[rel].clear()
                rels.clear()
            return active_rels, active_vars

        def hopcroft_karp(
            graph: dict[Relation, set[str]],
        ) -> tuple[dict[Relation, str], dict[str, Relation]]:
            match_rel: dict[Relation, str] = {}
            match_var: dict[str, Relation] = {}
            dist: dict[Relation, int] = {}
            inf = 10**9

            def bfs() -> bool:
                queue = deque()
                for rel in graph:
                    if rel not in match_rel:
                        dist[rel] = 0
                        queue.append(rel)
                    else:
                        dist[rel] = inf
                found = False
                while queue:
                    rel = queue.popleft()
                    for var in sorted(graph[rel]):
                        rel2 = match_var.get(var)
                        if rel2 is None:
                            found = True
                        elif dist.get(rel2, inf) == inf:
                            dist[rel2] = dist[rel] + 1
                            queue.append(rel2)
                return found

            def dfs(rel: Relation) -> bool:
                for var in sorted(graph[rel]):
                    rel2 = match_var.get(var)
                    if rel2 is None or (dist.get(rel2, inf) == dist[rel] + 1 and dfs(rel2)):
                        match_rel[rel] = var
                        match_var[var] = rel
                        return True
                dist[rel] = inf
                return False

            while bfs():
                for rel in graph:
                    if rel not in match_rel:
                        dfs(rel)
            return match_rel, match_var

        def tarjan_scc(
            nodes: Sequence[tuple[str, object]],
            graph: dict[tuple[str, object], list[tuple[str, object]]],
        ) -> list[list[tuple[str, object]]]:
            index = 0
            stack: list[tuple[str, object]] = []
            on_stack: set[tuple[str, object]] = set()
            indexes: dict[tuple[str, object], int] = {}
            lowlinks: dict[tuple[str, object], int] = {}
            result: list[list[tuple[str, object]]] = []

            def strongconnect(node: tuple[str, object]) -> None:
                nonlocal index
                indexes[node] = index
                lowlinks[node] = index
                index += 1
                stack.append(node)
                on_stack.add(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in indexes:
                        strongconnect(neighbor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                    elif neighbor in on_stack:
                        lowlinks[node] = min(lowlinks[node], indexes[neighbor])
                if lowlinks[node] == indexes[node]:
                    component: list[tuple[str, object]] = []
                    while True:
                        current = stack.pop()
                        on_stack.remove(current)
                        component.append(current)
                        if current == node:
                            break
                    result.append(component)

            for node in nodes:
                if node not in indexes:
                    strongconnect(node)
            return result

        def topo_order_sccs(
            sccs: Sequence[Sequence[tuple[str, object]]],
            graph: dict[tuple[str, object], list[tuple[str, object]]],
        ) -> list[list[tuple[str, object]]]:
            index_by_node: dict[tuple[str, object], int] = {}
            for idx, component in enumerate(sccs):
                for node in component:
                    index_by_node[node] = idx
            edges: dict[int, set[int]] = {idx: set() for idx in range(len(sccs))}
            indegree = [0] * len(sccs)
            for src, dests in graph.items():
                for dst in dests:
                    src_idx = index_by_node[src]
                    dst_idx = index_by_node[dst]
                    if src_idx == dst_idx or dst_idx in edges[src_idx]:
                        continue
                    edges[src_idx].add(dst_idx)
                    indegree[dst_idx] += 1
            queue = deque(sorted(idx for idx, deg in enumerate(indegree) if deg == 0))
            ordered: list[list[tuple[str, object]]] = []
            while queue:
                idx = queue.popleft()
                ordered.append(list(sccs[idx]))
                for dst_idx in sorted(edges[idx]):
                    indegree[dst_idx] -= 1
                    if indegree[dst_idx] == 0:
                        queue.append(dst_idx)
            return ordered

        def build_coupled_blocks(
            max_unknowns: int,
            max_relations: int,
        ) -> list[tuple[list[Relation], set[str]]]:
            rel_to_vars, var_to_rels = build_relation_graph()
            if not rel_to_vars:
                return []
            active_rels, active_vars = peel_leaf_pairs(rel_to_vars, var_to_rels)
            if not active_rels or not active_vars:
                return []
            graph: dict[Relation, set[str]] = {}
            for rel in active_rels:
                vars_for_rel = rel_to_vars.get(rel, set())
                vars_for_rel = {var for var in vars_for_rel if var in active_vars}
                if vars_for_rel:
                    graph[rel] = vars_for_rel
            if not graph:
                return []
            match_rel, match_var = hopcroft_karp(graph)
            nodes: set[tuple[str, object]] = set()
            directed: dict[tuple[str, object], list[tuple[str, object]]] = {}
            for rel, vars_for_rel in graph.items():
                rel_node = ("rel", rel)
                nodes.add(rel_node)
                for var in vars_for_rel:
                    var_node = ("var", var)
                    nodes.add(var_node)
                    if match_rel.get(rel) == var:
                        directed.setdefault(var_node, []).append(rel_node)
                    else:
                        directed.setdefault(rel_node, []).append(var_node)
            for node in nodes:
                directed.setdefault(node, [])

            def node_key(node: tuple[str, object]) -> tuple[int, int | str]:
                kind, payload = node
                if kind == "rel":
                    return (0, id(payload))
                return (1, payload)

            ordered_nodes = sorted(nodes, key=node_key)
            sccs = tarjan_scc(ordered_nodes, directed)
            ordered_sccs = topo_order_sccs(sccs, directed)

            blocks: list[tuple[list[Relation], set[str]]] = []
            for component in ordered_sccs:
                component_rels = [node[1] for node in component if node[0] == "rel"]
                component_vars = {node[1] for node in component if node[0] == "var"}
                if not component_rels or not component_vars:
                    continue
                if len(component_vars) > max_unknowns or len(component_rels) > max_relations:
                    continue
                blocks.append((component_rels, component_vars))
            return blocks

        def solve_coupled_blocks(
            max_unknowns: int,
            max_relations: int,
            *,
            allow_nsolve: bool,
        ) -> set[str]:
            updated_vars: set[str] = set()
            for component_rels, component_vars in build_coupled_blocks(max_unknowns, max_relations):
                updated_vars.update(
                    attempt_solve(component_rels, component_vars, allow_nsolve=allow_nsolve)
                )
            return updated_vars

        def solve_variable_systems(
            max_unknowns: int,
            max_relations: int,
            *,
            allow_nsolve: bool = False,
        ) -> set[str]:
            updated_vars: set[str] = set()
            for target in sorted(allowed_unknowns):
                if self.lock_explicit and target in explicit_names:
                    continue
                val = values.get(target, symbols[target])
                if numeric_value(val) is not None:
                    continue
                candidates: list[tuple[int, int, Relation, tuple[str, ...]]] = []
                for rel in rels_by_var.get(target, ()):
                    unknowns = relation_unknowns(rel)
                    if unknowns is None:
                        continue
                    candidates.append((len(unknowns), len(rel.variables), rel, unknowns))
                if not candidates:
                    continue
                candidates.sort(key=lambda item: (item[0], item[1]))
                if len(candidates) > max_relations:
                    candidates = candidates[:max_relations]
                component_rels: list[Relation] = []
                component_vars: set[str] = set()
                for _, _, rel, unknowns in candidates:
                    component_rels.append(rel)
                    component_vars.update(unknowns)
                if not component_vars or len(component_vars) > max_unknowns:
                    continue
                if len(component_rels) < len(component_vars):
                    continue
                updated_vars.update(
                    attempt_solve(component_rels, component_vars, allow_nsolve=allow_nsolve)
                )
            return updated_vars

        if global_mode:
            coupled_unknowns = 6
            coupled_relations = 8
            coupled_nsolve = True
        else:
            coupled_unknowns = 2
            coupled_relations = 3
            coupled_nsolve = False
        rels_by_var: dict[str, list[Relation]] = {}
        for rel in self.relations:
            if not rel.variables:
                continue
            for var in rel.variables:
                rels_by_var.setdefault(var, []).append(rel)

        queue = deque(rel for rel in self.relations if rel.variables)
        queued = set(queue)

        for _ in range(max_iterations):
            updated = False
            while queue:
                rel = queue.popleft()
                queued.discard(rel)
                updated_vars = process_relation(rel)
                if not updated_vars:
                    continue
                updated = True
                for var in updated_vars:
                    for rel_dep in rels_by_var.get(var, ()):
                        if rel_dep in queued:
                            continue
                        queue.append(rel_dep)
                        queued.add(rel_dep)
            if updated:
                continue
            coupled_updates = solve_coupled_blocks(
                max_unknowns=coupled_unknowns,
                max_relations=coupled_relations,
                allow_nsolve=coupled_nsolve,
            )
            if not coupled_updates and global_mode:
                coupled_updates = solve_variable_systems(
                    max_unknowns=coupled_unknowns,
                    max_relations=coupled_relations,
                    allow_nsolve=coupled_nsolve,
                )
            if not coupled_updates:
                break
            for var in coupled_updates:
                for rel_dep in rels_by_var.get(var, ()):
                    if rel_dep in queued:
                        continue
                    queue.append(rel_dep)
                    queued.add(rel_dep)

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
