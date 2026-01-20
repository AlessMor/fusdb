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
    ) -> None:
        self.relations = tuple(relations)
        self.rel_tol = rel_tol
        self.warn_sink = warn
        self._solve_cache: dict[tuple[int, str], list[tuple[tuple[str, ...], Callable[..., object] | None, sp.Expr]]] = {}

        # Build the relation-variable incidence graph once for component solving.
        graph = nx.Graph()
        for rel in self.relations:
            graph.add_node(rel, bipartite=0)
            for name in rel.variables:
                graph.add_node(name, bipartite=1)
                graph.add_edge(rel, name)

        components: list[tuple[list[Relation], list[str]]] = []
        for comp in nx.connected_components(graph):
            rels = [node for node in comp if isinstance(node, Relation)]
            if not rels:
                continue
            vars_ = [node for node in comp if isinstance(node, str)]
            # Keep a stable order for deterministic solving.
            components.append((sorted(rels, key=lambda r: r.name), sorted(vars_)))
        self._components = components

    def solve(
        self,
        values: Mapping[str, float] | None = None,
        *,
        mode: str = "override_input",
        tol: float | None = None,
        max_iter: int = 50,
        explicit: set[str] | None = None,
    ) -> dict[str, float]:
        """Solve the system using provided values as inputs/guesses.

        Args:
            values: Initial numeric values (explicit inputs and/or seeds).
            mode: "override_input" to allow overrides, "locked_input" to lock explicit inputs.
            tol: Global relative tolerance; defaults to system rel_tol.
            max_iter: Iteration cap for fills/repairs and numeric solve budget.
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

        # Pass 1: fill single-unknown relations and solve DM blocks on missing-only vars.
        for rels, vars_ in self._components:
            self._solve_component(rels, vars_, values_num, locked, max_iter, free_all=False)

        # Pass 2 (override_input): allow moving known values to satisfy relations.
        if override_input and self._validate(values_num, tol_use):
            for rels, vars_ in self._components:
                self._solve_component(rels, vars_, values_num, set(), max_iter, free_all=True)

        # Optional repair loop to satisfy violated relations by solving one variable.
        if override_input:
            self._repair_violations(values_num, tol_use, max_iter)

        # Final validation / error reporting.
        violations = self._validate(values_num, tol_use)
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
            for rel, detail in violations:
                msg = f"{rel.name} violates relation ({detail})"
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

    def _candidate_for_target(
        self,
        rel: Relation,
        target: str,
        values: Mapping[str, float],
    ) -> float | None:
        # Prefer cached symbolic solve plans when available.
        plan = self._solve_plan_for(rel, target)
        candidates: list[float] = []

        # Evaluate all viable expressions for the target and pick a candidate.
        for arg_names, fn, expr in plan:
            if any(name not in values for name in arg_names):
                continue
            args = [values[name] for name in arg_names]
            numeric = None
            if fn is not None:
                try:
                    numeric = float(fn(*args))
                except Exception:
                    numeric = None
            if numeric is None:
                subs = {symbol(name): sp.Float(values[name]) for name in arg_names}
                numeric = numeric_value(expr.subs(subs))
            if numeric is None or not math.isfinite(numeric):
                continue
            test_values = dict(values)
            test_values[target] = float(numeric)
            if not constraints_ok(rel.constraints_compiled, test_values, focus_names={target}):
                continue
            candidates.append(float(numeric))

        # Fallback: numeric solve for single-unknown relations with no symbolic plan.
        if not candidates:
            target_sym = symbol(target)
            subs = {
                symbol(name): sp.Float(values[name])
                for name in rel.variables
                if name != target and name in values
            }
            expr = rel.expr.subs(subs)
            if expr.free_symbols == {target_sym}:
                guesses = self._build_guesses((rel,), (target,), values)
                solved = solve_numeric_system([expr], [target_sym], guesses, max_iter=50)
                if solved:
                    numeric = float(solved[0])
                    if math.isfinite(numeric):
                        test_values = dict(values)
                        test_values[target] = numeric
                        if constraints_ok(rel.constraints_compiled, test_values, focus_names={target}):
                            candidates.append(numeric)

        if not candidates:
            return None
        return next((value for value in candidates if value > 0), candidates[0])

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

    def _solve_component(
        self,
        rels: Sequence[Relation],
        vars_: Sequence[str],
        values: dict[str, float],
        locked: set[str],
        max_iter: int,
        *,
        free_all: bool,
    ) -> None:
        # Inline single-unknown propagation within the component.
        for _ in range(max_iter):
            updated = False
            for rel in rels:
                # Only solve relations with exactly one missing value.
                missing = [name for name in rel.variables if name not in values]
                if len(missing) != 1:
                    continue
                target = missing[0]
                if target in locked:
                    continue
                if target not in rel.solve_targets:
                    continue
                # Compute a candidate for the missing variable.
                candidate = self._candidate_for_target(rel, target, values)
                if candidate is None:
                    continue
                if update_value(values, target, candidate, eps=_EPS):
                    updated = True
            if not updated:
                break

        if free_all:
            free_vars = set(vars_) - locked
        else:
            free_vars = {name for name in vars_ if name not in values and name not in locked}
        if not free_vars:
            return
        # Build relation -> unknown variables mapping for DM decomposition.
        rel_unknowns: dict[Relation, tuple[str, ...]] = {}
        for rel in rels:
            unknowns = tuple(
                name
                for name in rel.variables
                if name in free_vars and name in rel.solve_targets
            )
            if unknowns:
                rel_unknowns[rel] = unknowns
        if not rel_unknowns:
            return

        # Solve DM blocks in order, updating values as each block is resolved.
        for block_rels, block_vars in self._dm_blocks(rel_unknowns, free_vars):
            self._solve_block(block_rels, block_vars, values, max_iter)

    def _dm_blocks(
        self,
        rel_unknowns: Mapping[Relation, tuple[str, ...]],
        free_vars: set[str],
    ) -> list[tuple[list[Relation], list[str]]]:
        edges = [(rel, name) for rel, names in rel_unknowns.items() for name in names]
        if not edges:
            return []

        # Build incidence graph and compute maximum matching.
        rels = list(rel_unknowns.keys())
        graph = nx.Graph()
        graph.add_nodes_from(rels, bipartite=0)
        graph.add_nodes_from(free_vars, bipartite=1)
        graph.add_edges_from(edges)

        matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=set(rels))

        # Orient edges: matched var->rel, unmatched rel->var (DM convention).
        directed = nx.DiGraph()
        directed.add_nodes_from(graph.nodes())
        for rel, name in edges:
            matched = matching.get(rel) == name or matching.get(name) == rel
            directed.add_edge(name, rel) if matched else directed.add_edge(rel, name)

        unmatched_rels = [rel for rel in rels if rel not in matching]
        unmatched_vars = [name for name in free_vars if name not in matching]

        over = set()
        for rel in unmatched_rels:
            over.add(rel)
            over.update(nx.descendants(directed, rel))

        under = set()
        for name in unmatched_vars:
            under.add(name)
            under.update(nx.descendants(directed, name))

        # Retain only well-determined nodes, then order SCC blocks topologically.
        well_nodes = set(directed.nodes()) - over - under
        if not well_nodes:
            return []

        subgraph = directed.subgraph(well_nodes).copy()
        sccs = list(nx.strongly_connected_components(subgraph))
        condensed = nx.condensation(subgraph, sccs)

        def block_label(idx: int) -> tuple[str, ...]:
            members = condensed.nodes[idx]["members"]
            labels = []
            for node in members:
                if isinstance(node, Relation):
                    labels.append(f"R:{node.name}")
                else:
                    labels.append(f"V:{node}")
            return tuple(sorted(labels))

        blocks: list[tuple[list[Relation], list[str]]] = []
        for idx in nx.lexicographical_topological_sort(condensed, key=block_label):
            members = condensed.nodes[idx]["members"]
            block_rels = sorted(
                [node for node in members if isinstance(node, Relation)],
                key=lambda r: r.name,
            )
            block_vars = sorted([node for node in members if isinstance(node, str)])
            if not block_rels or not block_vars:
                continue
            # Skip underdetermined blocks and ones that depend on outside vars.
            if len(block_rels) < len(block_vars):
                continue
            block_var_set = set(block_vars)
            if any(
                not set(rel_unknowns.get(rel, ())).issubset(block_var_set)
                for rel in block_rels
            ):
                continue
            blocks.append((block_rels, block_vars))
        return blocks

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

    def _repair_violations(
        self,
        values: dict[str, float],
        tol: float,
        max_iter: int,
    ) -> None:
        # Iterate until no relation can be repaired via a single-variable update.
        for _ in range(max_iter):
            updated = False
            for rel in self.relations:
                # Only consider relations that are fully specified.
                if any(name not in values for name in rel.variables):
                    continue
                residual = relation_residual(values, rel.variables, rel.residual_fn, rel.expr)
                if residual is None:
                    continue
                rel_tol = rel.rel_tol if rel.rel_tol is not None else tol
                scale = max(max(abs(values[name]) for name in rel.variables), 1.0)
                if abs(residual) <= rel_tol * scale:
                    continue
                target = rel.solve_targets[0]
                # Solve a single variable to reduce residual.
                candidate = self._candidate_for_target(rel, target, values)
                if candidate is None:
                    continue
                if update_value(values, target, candidate, eps=_EPS):
                    updated = True
            if not updated:
                break

    def _validate(
        self,
        values: Mapping[str, float],
        tol: float,
    ) -> list[tuple[Relation, str]]:
        violations: list[tuple[Relation, str]] = []
        for rel in self.relations:
            # Check for missing values before residual evaluation.
            missing = [name for name in rel.variables if name not in values]
            if missing:
                violations.append((rel, f"missing values for: {', '.join(missing)}"))
                continue
            # Residual + constraint validation with scaled tolerance.
            residual = relation_residual(values, rel.variables, rel.residual_fn, rel.expr)
            rel_tol = rel.rel_tol if rel.rel_tol is not None else tol
            scale = max(max(abs(values[name]) for name in rel.variables), 1.0)
            constraint_ok = constraints_ok(rel.constraints_compiled, values)
            if residual is None or abs(residual) > rel_tol * scale or not constraint_ok:
                detail_parts = []
                if residual is None:
                    detail_parts.append("residual unavailable")
                else:
                    detail_parts.append(f"residual={residual}")
                    detail_parts.append(f"tol={rel_tol * scale}")
                if not constraint_ok:
                    detail_parts.append("constraints violated")
                violations.append((rel, ", ".join(detail_parts)))
        return violations
