"""Relation solver for fusdb.

Core ideas:
- Relations express truths (zero-residual equations).
- Reactor inputs are observations: override_input may override them; locked_input locks them.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Mapping, Sequence

import networkx as nx
import sympy as sp
from sympy.core.relational import Relational

from fusdb.relation_util import (
    REL_TOL_DEFAULT,
    constraints_ok,
    constraints_for_vars,
    numeric_value,
    parse_constraint,
    relation_residual,
    solve_linear_system,
    solve_numeric_system,
    symbol,
    update_value,
)

WarnFunc = Callable[[str, type[Warning] | None], None]

_EPS = 1e-12


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
    ) -> None:
        # Store core relation metadata and normalize inputs.
        self.name = name
        self.variables = tuple(variables)
        self.expr = sp.sympify(expr)
        self.rel_tol = rel_tol
        self.solve_for = solve_for
        self.initial_guesses = initial_guesses
        self.constraints = tuple(constraints or ())
        # Precompute solve targets and symbols for reuse.
        self.solve_targets = self.solve_for or self.variables
        self.syms = tuple(symbol(name) for name in self.variables)
        # Lambdify the residual for fast numeric evaluation.
        try:
            self.residual_fn = sp.lambdify(self.syms, self.expr, "math")
        except Exception:
            self.residual_fn = None
        # Compile constraints (relation-level + variable-level) once.
        compiled: list[tuple[tuple[str, ...], Callable[..., object] | None, Relational]] = []
        for item in list(self.constraints) + list(constraints_for_vars(self.variables)):
            constraint = item if isinstance(item, Relational) else parse_constraint(item)
            names = tuple(sorted(sym.name for sym in constraint.free_symbols))
            try:
                fn = sp.lambdify([symbol(name) for name in names], constraint, "math")
            except Exception:
                fn = None
            compiled.append((names, fn, constraint))
        self.constraints_compiled = tuple(compiled)


class RelationSystem:
    """Solve a set of relations with explicit values in override_input or locked_input mode."""

    def __init__(
        self,
        relations: Sequence[Relation],
        *,
        rel_tol: float = REL_TOL_DEFAULT,
        warn: WarnFunc = warnings.warn,
        warnings_issued: set[str] | None = None,
    ) -> None:
        self.relations = tuple(relations)
        self.rel_tol = rel_tol
        self.warn_sink = warn
        self._solve_cache: dict[tuple[int, str], list[tuple[tuple[str, ...], Callable[..., object] | None, sp.Expr]]] = {}
        # Use external warnings set if provided, otherwise create new one
        self._warnings_issued: set[str] = warnings_issued if warnings_issued is not None else set()

    def solve(
        self,
        values: Mapping[str, float] | None = None,
        *,
        mode: str = "override_input",
        tol: float | None = None,
        max_iter: int = 50,
        explicit: set[str] | None = None,
    ) -> dict[str, float]:
        """Solve system iteratively: exhaust 1×1, then 2×2, then 3×3, etc.

        Args:
            values: Initial numeric values (explicit inputs and/or seeds).
            mode: "override_input" to allow overrides, "locked_input" to lock explicit inputs.
            tol: Global relative tolerance; defaults to system rel_tol.
            max_iter: Iteration cap for numeric solve budget.
            explicit: Subset of keys in values treated as explicit inputs.
        """
        # Normalize inputs and explicit set.
        values_num = {k: float(v) for k, v in (values or {}).items()}
        explicit_set = set(values_num) if explicit is None else set(explicit) & set(values_num)
        explicit_values = {name: values_num[name] for name in explicit_set}
        # Normalize solve mode and tolerances.
        mode = mode.lower().strip()
        if mode not in {"override_input", "locked_input"}:
            raise ValueError("mode must be 'override_input' or 'locked_input'")
        override_input = mode == "override_input"
        tol_use = self.rel_tol if tol is None else float(tol)
        locked = set() if override_input else explicit_set
        
        # Note: Do NOT clear _warnings_issued here - we want warnings to be deduplicated
        # across multiple solve() calls on the same RelationSystem instance

        # Pass 1: Solve with explicit inputs locked
        # Dynamically identify solvable relations based on current known values
        all_vars = set()
        for rel in self.relations:
            all_vars.update(rel.variables)
        
        # Temporarily suppress warnings during Pass 1 (we'll try Pass 2 if this fails)
        warnings_backup = self._warnings_issued.copy()
        suppress_warnings = override_input
        
        self._solve_iterative(list(self.relations), sorted(all_vars), values_num, locked, max_iter, suppress_warnings=suppress_warnings)

        # Pass 2 (override_input): Allow adjusting any value to satisfy relations
        if override_input:
            violations_pass1 = self._validate(values_num, tol_use, explicit_set)
            if violations_pass1:
                # Pass 1 failed, try Pass 2 without clearing warnings
                # (we keep the warnings set to avoid duplicates in final validation)
                self._solve_iterative(list(self.relations), sorted(all_vars), values_num, set(), max_iter, suppress_warnings=False)

        # Final validation / error reporting.
        violations = self._validate(values_num, tol_use, explicit_set)
        if violations:
            if not override_input:
                lines = ["Unable to satisfy relations with locked reactor inputs:"]
                for rel, detail in violations:
                    lines.append(f"- {rel.name}: {detail}")
                    explicit_vars = [name for name in rel.variables if name in explicit_set]
                    if explicit_vars:
                        details = ", ".join(f"{name}={explicit_values[name]}" for name in explicit_vars)
                        lines.append(f"  explicit inputs: {details}")
                    else:
                        lines.append("  explicit inputs: none")
                raise ValueError("\n".join(lines))
            # Issue warnings for violations, but with deduplication
            for rel, detail in violations:
                msg = detail
                if msg not in self._warnings_issued:
                    self._warnings_issued.add(msg)
                    self.warn_sink(msg, UserWarning)

        if override_input:
            # Warn about explicit inputs that were overridden beyond tolerance.
            for name, explicit_value in explicit_values.items():
                solved_value = values_num.get(name)
                if solved_value is None:
                    continue
                scale = max(abs(explicit_value), abs(solved_value), 1.0)
                delta = solved_value - explicit_value
                if abs(delta) > tol_use * scale:
                    message = (
                        f"Explicit {name} overridden: explicit={explicit_value}, solved={solved_value}, "
                        f"delta={delta}, tol={tol_use}"
                    )
                    self.warn_sink(message, UserWarning)

        return values_num

    def _solve_for_any_target(
        self,
        rel: Relation,
        target: str,
        values: Mapping[str, float],
        suppress_warnings: bool = False,
    ) -> float | None:
        """Solve relation for target bidirectionally (ignoring solve_for restrictions)."""
        # Try symbolic solve first
        target_sym = symbol(target)
        known_subs = {symbol(v): sp.Float(values[v]) for v in rel.variables if v != target and v in values}
        expr = rel.expr.subs(known_subs)
        
        # If expression is already in target only, solve it
        if expr.free_symbols == {target_sym} or not expr.free_symbols:
            try:
                solutions = sp.solve(expr, target_sym)
                if not isinstance(solutions, list):
                    solutions = [solutions]
                
                for sol in solutions:
                    if not sol.free_symbols:  # Fully evaluated
                        numeric = numeric_value(sol)
                        
                        if numeric is not None and math.isfinite(numeric):
                            # Validate constraints
                            test = dict(values)
                            test[target] = numeric
                            if not constraints_ok(rel.constraints_compiled, test, focus_names={target}):
                                # Warn about constraint violation
                                if not suppress_warnings:
                                    failed_constraints = []
                                    for names, fn, constraint in rel.constraints_compiled:
                                        if target in names:
                                            try:
                                                result = fn(*[test.get(n, 0) for n in names]) if fn else None
                                                if result is False or (isinstance(result, bool) and not result):
                                                    failed_constraints.append(str(constraint))
                                            except Exception:
                                                pass
                                    
                                    if failed_constraints:
                                        constraints_str = ", ".join(failed_constraints)
                                        # Identify input variables that caused this
                                        input_vars = [v for v in rel.variables if v != target and v in values]
                                        if input_vars:
                                            inputs_str = ", ".join(f"{v}={values[v]:.4g}" for v in input_vars)
                                            msg = (
                                                f"{rel.name}: cannot solve for {target} = {numeric:.4g} from inputs ({inputs_str}) "
                                                f"(violates constraints: {constraints_str})"
                                            )
                                        else:
                                            msg = (
                                                f"{rel.name}: cannot solve for {target} = {numeric:.4g} "
                                                f"(violates constraints: {constraints_str})"
                                            )
                                        if msg not in self._warnings_issued:
                                            self._warnings_issued.add(msg)
                                            self.warn_sink(msg, UserWarning)
                                continue
                            
                            # Prefer positive solutions
                            if numeric > 0:
                                return numeric
            except Exception:
                pass
        
        # Fallback: numeric rootfinding with constraint-guided initial guesses
        def residual_fn(x):
            test = dict(values)
            test[target] = x[0]
            res = relation_residual(test, rel.variables, rel.residual_fn, rel.expr)
            return [res if res is not None else 1e10]
        
        from scipy.optimize import root
        
        # Extract constraint bounds for the target variable to guide initial guesses
        lower_bound, upper_bound = -1e10, 1e10
        for names, fn, constraint in rel.constraints_compiled:
            if target in names and len(names) == 1:
                # Simple single-variable constraint like "x <= 1" or "x >= 0"
                constraint_str = str(constraint)
                if '<=' in constraint_str:
                    parts = constraint_str.split('<=')
                    if len(parts) == 2:
                        try:
                            if target in parts[0]:
                                upper_bound = min(upper_bound, float(parts[1].strip()))
                            else:
                                lower_bound = max(lower_bound, float(parts[0].strip()))
                        except ValueError:
                            pass
                elif '>=' in constraint_str:
                    parts = constraint_str.split('>=')
                    if len(parts) == 2:
                        try:
                            if target in parts[0]:
                                lower_bound = max(lower_bound, float(parts[1].strip()))
                            else:
                                upper_bound = min(upper_bound, float(parts[0].strip()))
                        except ValueError:
                            pass
        
        # Build initial guesses, using constraint bounds when available
        guesses = self._build_guesses([rel], [target], values)
        guess_list = [guesses[0]]
        
        # Add constraint-guided guesses
        if math.isfinite(lower_bound) and math.isfinite(upper_bound):
            # Use midpoint and quartiles if we have bounds
            guess_list.extend([
                (lower_bound + upper_bound) / 2,
                lower_bound + (upper_bound - lower_bound) * 0.25,
                lower_bound + (upper_bound - lower_bound) * 0.75,
            ])
        elif math.isfinite(lower_bound):
            guess_list.extend([lower_bound + 1, lower_bound + 10])
        elif math.isfinite(upper_bound):
            guess_list.extend([upper_bound - 1, upper_bound - 10])
        else:
            # No constraint bounds, use generic guesses
            guess_list.extend([1.0, 0.1, 10.0, 100.0])
        
        # Try guesses in order
        for guess_val in guess_list:
            if not math.isfinite(guess_val):
                continue
            result = root(residual_fn, [guess_val], method='hybr')
            if result.success:
                numeric = float(result.x[0])
                if math.isfinite(numeric):
                    test = dict(values)
                    test[target] = numeric
                    if constraints_ok(rel.constraints_compiled, test, focus_names={target}):
                        return numeric
        
        return None

    def _solve_iterative(
        self,
        rels: Sequence[Relation],
        vars_: Sequence[str],
        values: dict[str, float],
        locked: set[str],
        max_iter: int,
        suppress_warnings: bool = False,
    ) -> None:
        """Iteratively solve n×n subsystems: 1×1 until exhausted, then 2×2, then 3×3, etc.
        After any progress at size>1, restart from 1×1.
        """
        max_size = 7
        
        for outer_iter in range(max_iter):
            made_progress = False
            
            # Try each size level
            for size in range(1, max_size + 1):
                # Keep solving systems of this size until no more progress
                while True:
                    size_progress = self._solve_all_of_size(rels, vars_, values, locked, size, max_iter, suppress_warnings)
                    if not size_progress:
                        break
                    made_progress = True
                    
                    # If we made progress and size > 1, restart from 1×1
                    # (solving larger systems may have unlocked new 1×1 systems)
                    if size > 1:
                        break
                
                # If we solved any size>1 systems, restart from 1×1
                if made_progress and size > 1:
                    break
            
            # If no progress at any size level, we're done
            if not made_progress:
                break
    
    def _solve_all_of_size(
        self,
        rels: Sequence[Relation],
        vars_: Sequence[str],
        values: dict[str, float],
        locked: set[str],
        size: int,
        max_iter: int,
        suppress_warnings: bool = False,
    ) -> bool:
        """Find and solve all size×size subsystems. Returns True if any progress made."""
        # Find all unknowns for each relation
        rel_unknowns: dict[Relation, list[str]] = {}
        for rel in rels:
            unknowns = [v for v in rel.variables if v in vars_ and v not in values and v not in locked]
            if unknowns:
                rel_unknowns[rel] = unknowns
        
        if not rel_unknowns:
            return False
        
        # For 1×1 systems: find relations with exactly 1 unknown and solve them
        if size == 1:
            made_progress = False
            for rel, unknowns in rel_unknowns.items():
                if len(unknowns) == 1:
                    target = unknowns[0]
                    result = self._solve_for_any_target(rel, target, values, suppress_warnings)
                    if result is not None and math.isfinite(result):
                        values[target] = result
                        made_progress = True
            return made_progress
        
        # For size > 1: find ALL square subsystems (n relations, n unknowns)
        # Key insight: unknowns can appear in OTHER relations too - we don't require "closure"
        # Strategy: enumerate all n-relation subsets, check if they have exactly n combined unknowns
        
        made_progress = False
        all_rels = list(rel_unknowns.keys())
        
        # For small sizes, enumerate all combinations; for large sizes, use heuristics
        if size <= 3 and len(all_rels) <= 20:
            # Enumerate all n-relation combinations
            from itertools import combinations
            for rel_subset in combinations(all_rels, size):
                # Get all unknowns in this subset
                subset_unknowns = set()
                for rel in rel_subset:
                    subset_unknowns.update(rel_unknowns[rel])
                
                # Check if square: exactly n unknowns
                if len(subset_unknowns) == size:
                    # Solve this subsystem numerically
                    if self._solve_numeric_system(list(rel_subset), values, locked, max_iter):
                        made_progress = True
        else:
            # For larger sizes or many relations, fall back to connected components
            # (This is the old approach - still useful for large systems)
            graph = nx.Graph()
            for rel, unknowns in rel_unknowns.items():
                graph.add_node(rel, bipartite=0)
                for var in unknowns:
                    graph.add_node(var, bipartite=1)
                    graph.add_edge(rel, var)
            
            for component in nx.connected_components(graph):
                comp_rels = [n for n in component if isinstance(n, Relation)]
                comp_vars = [n for n in component if isinstance(n, str)]
                
                if len(comp_rels) == size and len(comp_vars) == size:
                    # Try to solve this subsystem (even if not "closed")
                    if self._solve_numeric_system(comp_rels, values, locked, max_iter):
                        made_progress = True
        
        return made_progress
    
    def _solve_numeric_system(
        self,
        rels: list[Relation],
        values: dict[str, float],
        locked: set[str],
        max_iter: int,
    ) -> bool:
        """Solve system numerically using scipy.optimize. Returns True if solved."""
        # Collect unknowns
        unknowns = []
        for rel in rels:
            for var in rel.variables:
                if var not in values and var not in locked and var not in unknowns:
                    unknowns.append(var)
        
        if not unknowns:
            return False
        
        # For 1×1, try solving for the single unknown using bidirectional solving
        if len(unknowns) == 1 and len(rels) == 1:
            var = unknowns[0]
            rel = rels[0]
            candidate = self._solve_for_any_target(rel, var, values)
            if candidate is not None:
                values[var] = candidate
                return True
            return False
        
        # For larger systems, use block solver
        self._solve_block(rels, unknowns, values, max_iter)
        # Check if all variables were solved
        return all(v in values for v in unknowns)

    def _solve_plan_for(
        self, rel: Relation, target: str
    ) -> list[tuple[tuple[str, ...], Callable[..., object] | None, sp.Expr]]:
        # Reuse cached solve plans when possible.
        key = (id(rel), target)
        plan = self._solve_cache.get(key)
        if plan is not None:
            return plan

        # Fast path: isolate target via coefficient if expression is affine in target.
        target_sym = symbol(target)
        plan = []
        coeff = rel.expr.coeff(target_sym)
        if coeff is not None and coeff != 0:
            rest = rel.expr - coeff * target_sym
            if not rest.has(target_sym):
                expr = -rest / coeff
                arg_names = tuple(sorted(sym.name for sym in expr.free_symbols if sym.name != target))
                try:
                    fn = sp.lambdify([symbol(name) for name in arg_names], expr, "math")
                except Exception:
                    fn = None
                plan.append((arg_names, fn, expr))
                self._solve_cache[key] = plan
                return plan

        # Fallback: symbolic solve for the target.
        try:
            solutions = sp.solve(rel.expr, target_sym)
        except Exception:
            solutions = []
        if isinstance(solutions, sp.Expr):
            solutions = [solutions]
        for expr in solutions:
            arg_names = tuple(sorted(sym.name for sym in expr.free_symbols if sym.name != target))
            try:
                fn = sp.lambdify([symbol(name) for name in arg_names], expr, "math")
            except Exception:
                fn = None
            plan.append((arg_names, fn, expr))
        self._solve_cache[key] = plan
        return plan

    def _solve_block(
        self,
        block_rels: Sequence[Relation],
        block_vars: Sequence[str],
        values: dict[str, float],
        max_iter: int,
    ) -> None:
        unknowns = list(block_vars)
        if len(unknowns) <= 0:
            return

        # Map unknowns to symbols for solving.
        unknown_syms = [symbol(name) for name in unknowns]
        unknown_set = set(unknown_syms)

        # Substitute known values into equations and collect solvable expressions.
        known_subs: dict[sp.Symbol, sp.Expr] = {}
        for rel in block_rels:
            for name in rel.variables:
                if name in block_vars:
                    continue
                if name in values:
                    known_subs[symbol(name)] = sp.Float(values[name])

        equations: list[sp.Expr] = []
        for rel in block_rels:
            expr = rel.expr.subs(known_subs)
            if expr == 0:
                continue
            if expr.free_symbols and not expr.free_symbols.issubset(unknown_set):
                continue
            if expr.free_symbols:
                equations.append(expr)

        if len(equations) < len(unknown_syms):
            return

        solution: dict[sp.Symbol, sp.Expr] | None = None

        # Try linear solve first.
        solution = solve_linear_system(equations, unknown_syms)

        # Fallback: numeric solve for small blocks.
        if solution is None and len(unknown_syms) <= 6:
            guesses = self._build_guesses(block_rels, unknowns, values)
            solution_values = solve_numeric_system(
                equations,
                unknown_syms,
                guesses,
                max_iter=max_iter,
            )
            if solution_values is not None and len(solution_values) == len(unknown_syms):
                solution = dict(zip(unknown_syms, map(sp.Float, solution_values)))

        if solution is None:
            return

        # Validate numeric solutions before applying.
        solved_values: dict[str, float] = {}
        for sym, expr in solution.items():
            if expr.free_symbols:
                return
            numeric = numeric_value(expr)
            if numeric is None or not math.isfinite(numeric):
                return
            solved_values[sym.name] = float(numeric)

        # Validate constraints before applying updates.
        focus_names = set(solved_values)
        for rel in block_rels:
            values_map: dict[str, float] = {}
            for name in rel.variables:
                if name in solved_values:
                    values_map[name] = solved_values[name]
                elif name in values:
                    values_map[name] = float(values[name])
            if not constraints_ok(rel.constraints_compiled, values_map, focus_names=focus_names):
                return

        # Apply updates if they differ meaningfully.
        for name, numeric in solved_values.items():
            update_value(values, name, numeric, eps=_EPS)

    def _build_guesses(
        self,
        block_rels: Sequence[Relation],
        unknowns: Sequence[str],
        values: Mapping[str, float],
    ) -> list[float]:
        # Seed guesses from current known values.
        guess_values: dict[str, float] = {
            name: float(values[name])
            for name in unknowns
            if name in values and math.isfinite(values[name])
        }

        # Start with current values, then fill in provided guessers.
        def context() -> dict[str, float]:
            ctx = {k: float(v) for k, v in values.items() if math.isfinite(v)}
            ctx.update(guess_values)
            return ctx

        for rel in block_rels:
            guesses = rel.initial_guesses
            if not guesses:
                continue
            ctx = context()
            for name, guesser in guesses.items():
                if name not in unknowns or name in guess_values:
                    continue
                try:
                    guess = guesser(ctx) if callable(guesser) else guesser
                except Exception:
                    continue
                if isinstance(guess, (int, float)) and math.isfinite(float(guess)):
                    guess_values[name] = float(guess)

        # Fill remaining unknowns with a neutral default.
        return [guess_values.get(name, 1.0) for name in unknowns]

    def _validate(
        self,
        values: Mapping[str, float],
        tol: float,
        explicit: set[str] | None = None,
    ) -> list[tuple[Relation, str]]:
        violations: list[tuple[Relation, str]] = []
        explicit_set = explicit or set()
        
        for rel in self.relations:
            # Check for missing values - skip validation if any missing
            # (missing values are expected and not errors)
            missing = [name for name in rel.variables if name not in values]
            if missing:
                continue
            # Residual + constraint validation with scaled tolerance.
            residual = relation_residual(values, rel.variables, rel.residual_fn, rel.expr)
            rel_tol = rel.rel_tol if rel.rel_tol is not None else tol
            scale = max(max(abs(values[name]) for name in rel.variables), 1.0)
            constraint_ok = constraints_ok(rel.constraints_compiled, values)
            if residual is None or abs(residual) > rel_tol * scale or not constraint_ok:
                # Generate detailed error message
                msg = self._format_violation_message(rel, values, residual, rel_tol * scale, constraint_ok, explicit_set)
                violations.append((rel, msg))
        return violations
    
    def _format_violation_message(
        self,
        rel: Relation,
        values: Mapping[str, float],
        residual: float | None,
        tol: float,
        constraint_ok: bool,
        explicit_set: set[str],
    ) -> str:
        """Format a detailed violation message showing which variable is inconsistent."""
        # Lazy import to avoid circular dependency
        from fusdb.reactor_util import ALLOWED_VARIABLES
        
        # Find the explicit (input) variable(s) in this relation
        explicit_vars = [name for name in rel.variables if name in explicit_set]
        
        if not explicit_vars:
            # No explicit inputs - show relation name and basic diagnostic
            if residual is None:
                return f"{rel.name}: residual unavailable"
            elif not constraint_ok:
                return f"{rel.name}: constraints violated (residual={residual:.3e}, tol={tol:.3e})"
            else:
                return f"{rel.name}: residual={residual:.3e} exceeds tolerance {tol:.3e}"
        
        # Focus on the first explicit variable as the "inconsistent" one
        inconsistent_var = explicit_vars[0]
        input_value = values[inconsistent_var]
        
        # Get unit from registry
        var_meta = ALLOWED_VARIABLES.get(inconsistent_var, {})
        unit = var_meta.get("default_unit", "dimensionless")
        
        # Try to compute what the value should be by solving for this variable
        computed_value = None
        try:
            # Create a copy without the inconsistent variable
            test_values = {k: v for k, v in values.items() if k != inconsistent_var}
            # Try to solve for the inconsistent variable
            target_sym = symbol(inconsistent_var)
            known_subs = {symbol(v): sp.Float(test_values[v]) for v in rel.variables if v != inconsistent_var and v in test_values}
            expr = rel.expr.subs(known_subs)
            if expr.free_symbols == {target_sym}:
                solutions = sp.solve(expr, target_sym)
                if solutions:
                    if isinstance(solutions, list):
                        sol = solutions[0]
                    else:
                        sol = solutions
                    if not sol.free_symbols:
                        computed_value = float(sol.evalf())
        except Exception:
            pass
        
        # Format message
        if residual is None:
            # Can't format without residual
            return f"{inconsistent_var} is inconsistent for {rel.name} relation (residual unavailable)"
        
        if computed_value is not None:
            msg = f"{inconsistent_var} is inconsistent for {rel.name} relation. input: {input_value:.3e} {unit}, got: {computed_value:.3e} {unit}, residual: {residual:.3e} {unit}"
        else:
            # Fallback if we can't compute expected value
            msg = f"{inconsistent_var} is inconsistent for {rel.name} relation. input: {input_value:.3e} {unit}, residual: {residual:.3e} {unit}, tol: {tol:.3e} {unit}"
        
        if not constraint_ok:
            msg += " (constraints violated)"
        
        return msg
