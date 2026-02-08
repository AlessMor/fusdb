"""RelationSystem class to solve interconnected relations."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
from typing import Iterable
import itertools
import json
import logging
import math
import sympy as sp

from .registry import (
    allowed_variable_constraints,
)
from .variable_class import Variable
from .relation_util import evaluate_constraints
from .utils import safe_float, within_tolerance, normalize_solver_mode, ensure_list

logger = logging.getLogger(__name__)

@dataclass
class RelationSystem:
    """Collection of relations and variables with a solver."""
    relations: list
    variables: list[Variable]
    mode: str = "overwrite"
    verbose: bool = False
    n_max: int = 4
    max_passes: int = 100
    default_rel_tol: float = 0.01
    _inconsistency_warned: set[str] = field(default_factory=set)  # Track variables we've warned about
    _log: logging.LoggerAdapter = field(init=False, repr=False)
    # Dict-based graph structures (variable/relations connectivity and metadata).
    # Example shapes:
    # - _vars: {"R": Variable(...), "a": Variable(...), "P_fus": None}
    # - _vars_to_rels: {"R": {relA, relB}, "a": {relC}, ...}
    # - _rels_to_vars: {relA: ("R", "a", "B0"), ...}
    _vars: dict[str, Variable | None] = field(init=False, default_factory=dict, repr=False)
    _vars_to_rels: dict[str, set[object]] = field(init=False, default_factory=dict, repr=False)
    _rels_to_vars: dict[object, tuple[str, ...]] = field(init=False, default_factory=dict, repr=False)
    _var_order: list[str] = field(init=False, default_factory=list, repr=False)
    _var_constraints_map: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict, repr=False)
    _rel_constraints_map: dict[object, tuple[str, ...]] = field(init=False, default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize maps and caches. Args: none. Returns: None."""
        # Per-instance logger with structured context.
        base_logger = logger.getChild(self.__class__.__name__)
        base_logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        self._log = logging.LoggerAdapter(
            base_logger,
            {
                "system_id": id(self),
                "mode": self.mode,
                "pass_id": None,
                "inconsistency_warned": 0,
            },
        )

        self.relations = ensure_list(
            self.relations,
            name="RelationSystem relations",
            item_desc="Relation objects",
        )
        rel_nodes = self.relations
        self.variables = ensure_list(
            self.variables,
            name="RelationSystem variables",
            item_desc="Variable objects",
        )
        raw_vars = self.variables
        provided_pairs = [(var.name, var) for var in raw_vars]
        provided_vars = raw_vars
        self._log.debug(
            "RelationSystem.__post_init__: mode=%s, n_relations=%s, n_variables=%s",
            self.mode,
            len(rel_nodes),
            len(provided_vars),
        )
        # Normalize mode and set flags
        self.mode = normalize_solver_mode(self.mode)
        self._log.debug(f"Normalized mode: {self.mode}")
        self.pass_id = -1
        self.overrides: dict[str, dict[str, object]] = {}
        self._log.extra["pass_id"] = self.pass_id
        self._log.extra["mode"] = self.mode
        self._log.extra["inconsistency_warned"] = len(self._inconsistency_warned)
        
        # Build dict-based bipartite structures (relations <-> variables).
        # We keep variable metadata/constraints separate from connectivity.
        seen_names: set[str] = set()
        var_by_name = {name: var for name, var in provided_pairs}
        self._var_order = [name for name, _ in provided_pairs]
        var_names: set[str] = set(self._var_order)

        self._rels_to_vars = {}
        self._vars_to_rels = {}
        for idx, rel in enumerate(rel_nodes):
            rel_name = getattr(rel, "name", None) or f"relation_{idx}"
            if rel_name in seen_names:
                logger.warning(
                    "Duplicate relation name '%s' detected (duplicate index %s).",
                    rel_name,
                    idx,
                )
            else:
                seen_names.add(rel_name)

            vars_for_rel = tuple(v for v in getattr(rel, "variables", ()) if v is not None)
            self._rels_to_vars[rel] = vars_for_rel
            for name in vars_for_rel:
                self._vars_to_rels.setdefault(name, set()).add(rel)
            var_names.update(vars_for_rel)

        # Ensure all provided variables appear in the adjacency map (even if isolated).
        for name in self._var_order:
            self._vars_to_rels.setdefault(name, set())

        var_names_sorted = sorted(var_names)
        for name in var_names_sorted:
            if name not in self._var_order:
                self._var_order.append(name)
        self._log.debug(f"Total variables in system: {len(var_names_sorted)}")

        var_with_constraints = 0
        for name in var_names_sorted:
            var = var_by_name.get(name)
            if var is not None and var.rel_tol is None and var.abs_tol is None:
                var.rel_tol = self.default_rel_tol
            cons = allowed_variable_constraints(name)
            if cons:
                var_with_constraints += 1
            self._vars[name] = var
            self._var_constraints_map[name] = cons

        rel_with_constraints = 0
        for rel in rel_nodes:
            cons = tuple(getattr(rel, "constraints", ()) or ())
            if cons:
                rel_with_constraints += 1
            self._rel_constraints_map[rel] = cons

        self._log.debug(
            "Built constraints: %s variables with constraints, %s relations with constraints",
            var_with_constraints,
            rel_with_constraints,
        )
        
        self._violated: set[object] = set()
        self._pending_rels: set[object] = set(rel_nodes)
        if self.mode == 'overwrite':
            self._start_pass()

    @property
    def variables_dict(self) -> dict[str, Variable]:
        """Return current variables keyed by name (dict graph is source of truth)."""
        vars_map: dict[str, Variable] = {}
        for name in self._var_order:
            var = self._vars.get(name)
            if var is not None:
                vars_map[name] = var
        return vars_map
    
    def _get_value(self, name: str) -> object | None:
        """Get the effective value for a variable, respecting pass_mode and overrides.
        
        This centralizes the complex logic for value retrieval that considers:
        - Pass mode: uses current pass_id to get values from the solving pass
        - Non-pass mode: uses latest computed values
        
        Args:
            name: Variable name to retrieve
        
        Returns:
            The effective value or None if not available
        """
        var = self._vars.get(name)
        if var is None:
            return None

        if self.mode == 'overwrite':
            # In pass mode, get value from current pass or fall back to input
            if self.pass_id >= 0:
                value = var.get_value_at_pass(self.pass_id)
                if value is not None:
                    self._log.debug(f"_get_value({name}): pass_id={self.pass_id}, value={value}")
                    return value
            input_val = var.input_value
            if input_val is None:
                self._log.debug(f"_get_value({name}): no value in pass {self.pass_id}, no input value")
            return input_val

        # In non-pass mode, get latest computed value
        for value in reversed(var.values):
            if value is not None:
                return value
        self._log.debug(f"_get_value({name}): no current value (non-pass mode)")
        return None

    def _values_dict(self) -> dict[str, object]:
        """Return mapping of effective values."""
        return {n: v for n in self._var_order if (v := self._get_value(n)) is not None}

    def _accept_candidate_values(
        self,
        solved: dict[str, object],
        *,
        rels: list[object],
        mode_overwrite: bool,
        reason: str,
        relation: str | list[str] | None,
        values_map: dict[str, object] | None = None,
        protect_explicit: bool = False,
        warn_input: bool = False,
        check_violation_increase: bool = False,
    ) -> bool:
        """Validate and commit candidate values in one place."""
        if not solved:
            return False

        base_values = dict(values_map or self._values_dict())
        merged = dict(base_values)
        merged.update(solved)
        for name in solved:
            if self._constraints_violated(merged, names=[name]):
                return False
        for rel in rels:
            if self._constraints_violated(merged, rel=rel, names=()):
                return False

        if check_violation_increase:
            if len(solved) != 1:
                return False
            name, value = next(iter(solved.items()))
            value_scalar = safe_float(value)
            if value_scalar is None:
                return False
            rels_to_check = self._vars_to_rels.get(name, ())
            if rels_to_check:
                base_violations = len(
                    self._violated_relations(base_values, list(rels_to_check))
                )
                candidate_values = dict(base_values)
                candidate_values[name] = value_scalar
                candidate_violations = len(
                    self._violated_relations(candidate_values, list(rels_to_check))
                )
                if candidate_violations > base_violations:
                    self._log.debug(
                        "    ✗ Candidate for %s increases violations",
                        name,
                    )
                    return False

        accepted: dict[str, object] = {}
        for name, value in solved.items():
            var = self._vars.get(name)
            if var is not None and var.fixed:
                return False

            value_scalar = safe_float(value)
            if value_scalar is None:
                return False

            cur_value = base_values.get(name)
            cur_scalar = safe_float(cur_value) if cur_value is not None else None
            rel_tol = var.rel_tol if var and var.rel_tol is not None else self.default_rel_tol
            abs_tol = var.abs_tol if var and var.abs_tol is not None else 0.0

            if warn_input and cur_scalar is not None:
                input_value = var.input_value if var is not None else None
                if input_value is not None and name not in self._inconsistency_warned:
                    input_scalar = safe_float(input_value)
                    if input_scalar is not None and not within_tolerance(
                        input_scalar, value_scalar, rel_tol=rel_tol, abs_tol=abs_tol
                    ):
                        rel_name = (
                            relation
                            if isinstance(relation, str)
                            else (relation[0] if relation else "unknown")
                        )
                        self._log.warning(
                            "Inconsistency: relation '%s' computed %s = %.3g, but input specifies %s = %.3g",
                            rel_name,
                            name,
                            value_scalar,
                            name,
                            input_scalar,
                        )
                        self._inconsistency_warned.add(name)
                        self._log.extra["inconsistency_warned"] = len(self._inconsistency_warned)

            if cur_scalar is not None and within_tolerance(
                cur_scalar, value_scalar, rel_tol=rel_tol, abs_tol=abs_tol
            ):
                continue

            if (
                protect_explicit
                and var is not None
                and var.input_source == "explicit"
                and cur_scalar is not None
            ):
                change = abs(value_scalar - cur_scalar) / max(abs(cur_scalar), 1.0)
                if change > (rel_tol or 0.0):
                    return False

            accepted[name] = value

        if not accepted:
            return False

        for name, value in accepted.items():
            self._set_value(
                name,
                value,
                overwrite=mode_overwrite,
                reason=reason,
                relation=relation,
            )
        return True

    
    def _set_value(
        self,
        name: str,
        value: object,
        *,
        overwrite: bool = False,
        reason: str | None = None,
        relation: str | list[str] | None = None,
        pass_id: int | None = None,
    ) -> None:
        """Store a solved value in values map and update caches/pending relations."""
        self._log.debug(f"_set_value({name}={value}, overwrite={overwrite}, pass_mode={self.mode == 'overwrite'}, reason={reason})")
        if pass_id is None and self.mode == 'overwrite':
            pass_id = self.pass_id
        var = self._vars.get(name)
        if var is None:
            self._log.debug(f"Creating new variable: {name}")
            var = Variable(name=name)
            if var.rel_tol is None and var.abs_tol is None:
                var.rel_tol = self.default_rel_tol
            if name not in self._var_order:
                self._var_order.append(name)
            self._vars[name] = var
            self._var_constraints_map.setdefault(name, allowed_variable_constraints(name))
            self._vars_to_rels.setdefault(name, set())
        changed = var.add_value(value, pass_id=pass_id, reason=reason, relation=relation, default_rel_tol=self.default_rel_tol)
        if not changed:
            if pass_id is None:
                self._log.debug(f"  No change for {name}")
            else:
                self._log.debug(f"  No change for {name} (value already set or within tolerance)")
            return
        rels: set[object] = set(self._vars_to_rels.get(name, ()))
        if rels:
            if pass_id is None:
                self._log.debug(f"  {name} updated, adding {len(rels)} relations to pending")
            else:
                affected_rels = len(rels)
                self._log.debug(f"  {name} affects {affected_rels} relations, adding to pending")
            self._pending_rels.update(rels)
    
    def _expected(self, rel: object, values: dict[str, object]) -> float | None:
        try:
            expected = rel.evaluate(**{name: values[name] for name in rel.inputs})
        except Exception:
            return None
        return safe_float(expected)

    def _residual(self, rel: object, values: dict[str, object], *, scaled: bool = False) -> float | None:
        expected_scalar = self._expected(rel, values)
        if expected_scalar is None:
            return None
        actual_scalar = safe_float(values.get(rel.output))
        if actual_scalar is None:
            return None
        residual = actual_scalar - expected_scalar
        if not scaled:
            return residual
        scale = max(abs(expected_scalar), abs(actual_scalar), 1.0)
        return residual / scale

    def _residual_derivative(
        self,
        rel: object,
        name: str,
        values: dict[str, object],
        *,
        current: float | None = None,
    ) -> float | None:
        if name == rel.output:
            return 1.0
        if current is None:
            current = safe_float(values.get(name))
        if current is None:
            return None
        step = 1e-6 * max(abs(current), 1.0)
        v_plus = dict(values)
        v_minus = dict(values)
        v_plus[name] = current + step
        v_minus[name] = current - step
        f_plus = self._expected(rel, v_plus)
        f_minus = self._expected(rel, v_minus)
        if f_plus is None or f_minus is None:
            return None
        dfdx = (f_plus - f_minus) / (2 * step)
        if dfdx == 0 or not math.isfinite(dfdx):
            return None
        return -dfdx

    @staticmethod
    def _candidate_better(
        key: tuple[int, float, float],
        best_key: tuple[int, float, float] | None,
        *,
        tol: float = 1e-12,
    ) -> bool:
        if best_key is None:
            return True
        c_viol, c_sse, c_change = key
        b_viol, b_sse, b_change = best_key
        if c_viol < b_viol:
            return True
        if c_viol == b_viol:
            if c_sse < b_sse - tol:
                return True
            if abs(c_sse - b_sse) <= tol and c_change < b_change - tol:
                return True
        return False

    def _constraints_violated(
        self,
        values: dict[str, object],
        *,
        rel: object | None = None,
        names: Iterable[str] | None = None,
    ) -> bool:
        if rel is not None and evaluate_constraints(self._rel_constraints_map.get(rel, ()), values):
            return True
        if names is None:
            names = self._rels_to_vars.get(rel, ()) if rel is not None else ()
        return any(
            evaluate_constraints(self._var_constraints_map.get(name, ()), values)
            for name in names
        )


    def _solve_for_value(
        self,
        rel: object,
        name: str,
        values_map: dict[str, object],
        *,
        prefer_eval_output: bool = False,
    ) -> object | None:
        if prefer_eval_output and name == rel.output and all(n in values_map for n in rel.inputs):
            try:
                return rel.evaluate(**{n: values_map[n] for n in rel.inputs})
            except Exception:
                return None
        try:
            solved = rel.solve_for_value(name, values_map)
        except Exception:
            solved = None
        if solved is None and name == rel.output:
            try:
                solved = rel.evaluate(**{n: values_map[n] for n in rel.inputs})
            except Exception:
                return None
        return solved


    def _apply_relation(
        self,
        rel: object,
        rel_values: dict[str, object],
        missing_inputs: list[str],
        *,
        mode_overwrite: bool,
    ) -> bool:
        """Apply a relation forward or backward; return True if a value was set."""
        if missing_inputs:
            self._log.debug(
                "  Relation '%s': %s missing inputs %s",
                rel.name,
                len(missing_inputs),
                missing_inputs,
            )
            output_value = rel_values.get(rel.output)
            if len(missing_inputs) != 1 or output_value is None:
                return False
            missing_var = missing_inputs[0]
            self._log.debug("    Attempting backward solve for %s", missing_var)
            if rel.solve_for and missing_var not in rel.solve_for:
                self._log.debug(
                    "    Skipping backward solve for %s (not allowed by solve_for)",
                    missing_var,
                )
                return False

            known_values = {
                name: value
                for name, value in rel_values.items()
                if name != missing_var and value is not None
            }
            try:
                solved_value = rel.solve_for_value(missing_var, known_values)
            except Exception as e:
                self._log.debug("    ✗ Backward solve failed: %s", e)
                self._log.info(
                    "Backward solve failed for %s in '%s': %s",
                    missing_var,
                    rel.name,
                    e,
                )
                return False

            if solved_value is None:
                self._log.debug("    ✗ Backward solve returned None")
                return False
            solved_scalar = safe_float(solved_value)
            if solved_scalar is None:
                self._log.debug("    ✗ Backward solve returned non-finite value")
                return False

            if not self._accept_candidate_values(
                {missing_var: solved_scalar},
                rels=[rel],
                mode_overwrite=mode_overwrite,
                reason="relation_inverse",
                relation=rel.name,
                protect_explicit=False,
                warn_input=False,
                check_violation_increase=True,
            ):
                self._log.debug("    ✗ Backward solve rejected for %s", missing_var)
                return False

            self._log.debug("    ✓ Backward solved %s = %.6g", missing_var, solved_scalar)
            self._log.info(
                "Solved backwards: %s = %.4g from '%s'",
                missing_var,
                solved_scalar,
                rel.name,
            )
            return True

        out = rel.output
        inputs_map = {name: rel_values[name] for name in rel.inputs}
        try:
            res = rel.evaluate(**inputs_map)
            self._log.debug("  Relation '%s': evaluated %s = %s", rel.name, out, res)
        except Exception as e:
            self._log.debug("  Relation '%s': evaluation failed: %s", rel.name, e)
            return False

        res_s = safe_float(res)
        if res_s is None:
            return False

        if not self._accept_candidate_values(
            {out: res},
            rels=[rel],
            mode_overwrite=mode_overwrite,
            reason="relation",
            relation=rel.name,
            protect_explicit=True,
            warn_input=True,
            check_violation_increase=False,
        ):
            self._log.debug("  Relation '%s': candidate rejected", rel.name)
            return False

        self._log.debug("  Relation '%s': set %s = %s", rel.name, out, res_s)
        return True

    def _infer_var_bounds(self, name: str) -> tuple[float | None, float | None]:
        """Infer simple numeric bounds from variable constraints, if possible."""
        lower: float | None = None
        upper: float | None = None

        for constraint in self._var_constraints_map.get(name, ()):
            try:
                expr = sp.sympify(constraint)
            except Exception:
                continue
            for arg in (expr.args if isinstance(expr, sp.And) else (expr,)):
                if not hasattr(arg, "rel_op"):
                    continue
                lhs, rhs = arg.lhs, arg.rhs
                lhs_is_name = isinstance(lhs, sp.Symbol) and lhs.name == name
                rhs_is_name = isinstance(rhs, sp.Symbol) and rhs.name == name
                try:
                    lhs_num = float(lhs) if getattr(lhs, "is_number", False) else None
                    rhs_num = float(rhs) if getattr(rhs, "is_number", False) else None
                except Exception:
                    lhs_num = None
                    rhs_num = None
                op = arg.rel_op
                if op in (">", ">="):
                    if lhs_is_name and rhs_num is not None:
                        lower = rhs_num if lower is None else max(lower, rhs_num)
                    elif rhs_is_name and lhs_num is not None:
                        upper = lhs_num if upper is None else min(upper, lhs_num)
                elif op in ("<", "<="):
                    if lhs_is_name and rhs_num is not None:
                        upper = rhs_num if upper is None else min(upper, rhs_num)
                    elif rhs_is_name and lhs_num is not None:
                        lower = lhs_num if lower is None else max(lower, lhs_num)
        return lower, upper

    def _constraint_residuals(
        self,
        constraints: tuple[str, ...],
        values: dict[str, object],
        *,
        penalty: float,
    ) -> list[float]:
        """Return per-constraint penalty residuals (0 if satisfied)."""
        if not constraints:
            return []
        residuals: list[float] = []
        for constraint in constraints:
            try:
                expr = sp.sympify(constraint, locals=values)
                if expr is sp.S.true:
                    residuals.append(0.0)
                elif expr is sp.S.false:
                    residuals.append(penalty)
                else:
                    residuals.append(0.0 if bool(expr) else penalty)
            except Exception:
                residuals.append(penalty)
        return residuals

    def _least_squares_block_compact(
        self,
        relations: list[object],
        unknowns: list[str],
        values_map: dict[str, object],
    ) -> dict[str, object] | None:
        """Compact nonlinear least-squares block solve (no domain transforms)."""
        try:
            from scipy.optimize import least_squares
        except Exception:
            return None
        if len(relations) != len(unknowns):
            return None
        penalty = 1e3

        def F(x: list[float]) -> list[float]:
            merged = dict(values_map)
            merged.update(dict(zip(unknowns, x)))
            res: list[float] = []
            for rel in relations:
                r = self._residual(rel, merged, scaled=True)
                res.append(r if r is not None and math.isfinite(r) else penalty)
            for name in unknowns:
                res += self._constraint_residuals(self._var_constraints_map.get(name, ()), merged, penalty=penalty)
            for rel in relations:
                res += self._constraint_residuals(self._rel_constraints_map.get(rel, ()), merged, penalty=penalty)
            return res

        bounds = [self._infer_var_bounds(name) for name in unknowns]
        lower = [(-math.inf if lo is None else lo) for lo, _ in bounds]
        upper = [(math.inf if hi is None else hi) for _, hi in bounds]

        base: list[float] = []
        for name in unknowns:
            scalar = safe_float(values_map.get(name))
            if scalar is None:
                for rel in relations:
                    guess_fn = rel.initial_guesses.get(name) if rel.initial_guesses else None
                    if guess_fn is None:
                        continue
                    try:
                        guess = guess_fn(values_map)
                    except Exception:
                        continue
                    scalar = safe_float(guess)
                    if scalar is not None:
                        break
            base.append(scalar if scalar is not None else 1.0)
        best = None
        best_cost = None
        for scale in (1e-3, 1e-1, 1.0, 1e1, 1e3):
            x0 = []
            for guess, lo, hi in zip(base, lower, upper):
                val = guess * scale
                if not math.isfinite(val):
                    val = guess
                if math.isfinite(lo):
                    val = max(val, lo)
                if math.isfinite(hi):
                    val = min(val, hi)
                x0.append(val)
            try:
                r = least_squares(
                    F,
                    x0,
                    method="trf",
                    loss="soft_l1",
                    x_scale="jac",
                    max_nfev=200,
                    bounds=(lower, upper),
                )
            except Exception:
                continue
            if not r.success:
                continue
            if best_cost is None or r.cost < best_cost:
                best_cost = r.cost
                best = r.x
        return dict(zip(unknowns, best)) if best is not None else None

    def _solve_block(
        self,
        relations: list[object],
        unknowns: list[str],
        values_map: dict[str, object],
    ) -> dict[str, object] | None:
        """Solve a square block (nxn) of relations for unknowns."""
        if len(relations) != len(unknowns):
            return None

        for rel in relations:
            if set([*rel.inputs, rel.output]).issubset(set(unknowns)):
                return None

        if len(unknowns) == 1 and len(relations) == 1:
            rel = relations[0]
            unknown = unknowns[0]
            solved = self._solve_for_value(rel, unknown, values_map, prefer_eval_output=True)
            return None if solved is None else {unknown: solved}

        return self._least_squares_block_compact(relations, unknowns, values_map)

    def _build_unknown_map(
        self,
        rel_nodes: list[object],
        values: dict[str, object],
    ) -> dict[frozenset[str], list[object]]:
        """Group relations by their unknown variable sets."""
        unknown_map: dict[frozenset[str], list[object]] = {}
        # unknown_map example: {frozenset({"R", "a"}): [rel1, rel2], ...}
        for rel in rel_nodes:
            unknowns = frozenset(n for n in self._rels_to_vars.get(rel, ()) if n not in values)
            unknown_map.setdefault(unknowns, []).append(rel)
        return unknown_map

    def _solve_unknown_blocks(
        self,
        unknown_map: dict[frozenset[str], list[object]],
        values: dict[str, object],
        *,
        mode_overwrite: bool,
    ) -> bool:
        """Try solving square blocks (1x1..n_max) from grouped unknowns."""
        self._log.debug(f"_solve_unknowns: trying block sizes 1..{self.n_max}")
        for size in range(1, self.n_max + 1):
            candidates = [
                (m, r)
                for m, r in unknown_map.items()
                if m and len(m) == size and len(r) >= size
            ]
            self._log.info("%sx%s blocks: %s", size, size, len(candidates))
            self._log.debug(f"  {size}x{size} blocks: {len(candidates)} candidates")
            for unknown_set, rels in candidates:
                unknowns = sorted(unknown_set)
                rels_sorted = list(rels)
                self._log.debug(f"    Block with unknowns {unknowns}, {len(rels)} relations")
                for rel_subset in itertools.combinations(rels_sorted, size):
                    rel_names = [rel.name for rel in rel_subset]
                    self._log.debug(f"      Trying relations: {rel_names}")
                    if size == 1:
                        rel = rel_subset[0]
                        unknown = unknowns[0]
                        if rel.solve_for and unknown not in rel.solve_for:
                            continue
                    solved = self._solve_block(list(rel_subset), unknowns, values)
                    if solved and self._accept_candidate_values(
                        solved,
                        rels=list(rel_subset),
                        mode_overwrite=mode_overwrite,
                        reason="solve",
                        relation=rel_names,
                        values_map=values,
                        protect_explicit=False,
                        warn_input=False,
                        check_violation_increase=False,
                    ):
                        self._log.debug("      ✓ Solved: %s", solved)
                        self._log.info(
                            "Solved block %s from relations %s",
                            unknowns,
                            ", ".join(rel_names),
                        )
                        return True
                    self._log.debug("      ✗ Failed to solve or constraints violated")
        self._log.debug(f"_solve_unknowns: no blocks solved")
        return False

    def _enforce_pending_relations(
        self,
        rel_index: dict[object, int],
        *,
        mode_overwrite: bool,
    ) -> tuple[int, set[object]]:
        """Apply forward/backward relation solves until no pending relations remain."""
        pending = deque(sorted(self._pending_rels, key=rel_index.get))
        self._pending_rels.clear()
        self._log.debug(f"_enforce_relations: processing {len(pending)} pending relations")
        applied = 0
        steps = 0
        iterations = 0
        touched: set[object] = set()

        while pending:
            rel = pending.popleft()
            touched.add(rel)
            steps += 1

            # Gather values for this relation once.
            rel_values = {name: self._get_value(name) for name in self._rels_to_vars.get(rel, ())}
            missing_inputs = [name for name in rel.inputs if rel_values.get(name) is None]

            if self._apply_relation(
                rel,
                rel_values,
                missing_inputs,
                mode_overwrite=mode_overwrite,
            ):
                applied += 1

            # If new values were set, rebuild pending queue in relation order.
            if not pending:
                if not self._pending_rels:
                    break
                pending = deque(sorted(self._pending_rels, key=rel_index.get))
                self._pending_rels.clear()
                iterations += 1
                if iterations > 100:
                    logger.warning(f"_execute_pass: breaking after {iterations} iterations")
                    break

        if applied:
            self._log.info("Applied %s relation outputs", applied)
            self._log.debug(f"_enforce_relations: applied {applied} relations")
        self._log.debug(f"_execute_pass: processed {steps} relations")
        return applied, touched

    
    def _start_pass(self) -> None:
        """Start a new override pass and seed inputs. Args: none. Returns: None."""
        self.pass_id += 1
        self._pending_rels = set(self.relations)
        
        for name in self._var_order:
            var = self._vars.get(name)
            if var is None or var.input_source is None:
                continue
            
            override = self.overrides.get(name)
            base = override.get("value") if override else var.input_value
            if base is None:
                continue
            
            reason = "override" if override else "input"
            relation = override.get("relation") if override else None
            self._set_value(name, base, reason=reason, relation=relation)

    def _select_culprit(
        self,
        rels: set[object],
        values: dict[str, object],
        rel_nodes: list[object],
    ):
        base_violations = len(self._violated_relations(values))
        base_sse = 0.0
        for rel in rel_nodes:
            if any(name not in values for name in self._rels_to_vars.get(rel, ())):
                continue
            residual = self._residual(rel, values, scaled=True)
            if residual is None:
                continue
            base_sse += residual * residual

        candidates: list[tuple[str, float, float, str, float]] = []
        for rel in rels:
            culprit = self._culprit_for_relation(rel, values)
            if culprit:
                residual = self._residual(rel, values, scaled=True)
                severity = abs(residual) if residual is not None else 0.0
                candidates.append((*culprit, rel.name, severity))

        if not candidates:
            neighbors: set[object] = set()
            for rel in rels:
                for var in self._rels_to_vars.get(rel, ()):
                    for rel_neighbor in self._vars_to_rels.get(var, ()):
                        if rel_neighbor is None or rel_neighbor is rel:
                            continue
                        neighbors.add(rel_neighbor)
            for rel in neighbors:
                culprit = self._culprit_for_relation(rel, values)
                if culprit:
                    residual = self._residual(rel, values, scaled=True)
                    severity = abs(residual) if residual is not None else 0.0
                    candidates.append((*culprit, rel.name, severity))

        if candidates:
            candidates.sort(key=lambda item: (-item[4], item[1]))
            best_local = None
            best_key = None
            for name, change, target, rel_name, _severity in candidates:
                target_scalar = safe_float(target)
                if target_scalar is None:
                    continue
                var = self._vars.get(name)
                current = var.input_value if var is not None else None
                current_scalar = safe_float(current)
                if current_scalar is None:
                    continue
                rel_tol = var.rel_tol or 0.0 if var else 0.0
                abs_tol = var.abs_tol or 0.0 if var else 0.0
                if within_tolerance(current_scalar, target_scalar, rel_tol=rel_tol, abs_tol=abs_tol):
                    continue
                next_values = dict(values)
                next_values[name] = target_scalar
                violations_after = len(self._violated_relations(next_values))
                sse_after = 0.0
                for rel in rel_nodes:
                    if any(n not in next_values for n in self._rels_to_vars.get(rel, ())):
                        continue
                    residual = self._residual(rel, next_values, scaled=True)
                    if residual is None:
                        continue
                    sse_after += residual * residual
                key = (violations_after, sse_after, change)
                if self._candidate_better(key, best_key):
                    best_key = key
                    best_local = (name, change, target_scalar, rel_name, violations_after, sse_after)

            if best_local:
                name, change, target, rel_name, violations_after, sse_after = best_local
                if violations_after < base_violations or (
                    violations_after == base_violations and sse_after < base_sse
                ):
                    return ("local", name, change, target, rel_name, None)

        rel_list = list(rels)
        if not rel_list:
            return None

        explicit_vars = {
            name
            for name in self._var_order
            if (var := self._vars.get(name)) and var.input_source == "explicit" and var.input_value is not None
        }

        rel_data: list[tuple[object, float]] = []
        for rel in rel_list:
            if any(name not in values or values.get(name) is None for name in self._rels_to_vars.get(rel, ())):
                continue
            residual = self._residual(rel, values)
            if residual is None:
                continue
            rel_data.append((rel, residual))

        if not rel_data:
            return None

        sse_before = sum(residual * residual for _, residual in rel_data)
        best_global = None
        best_key = None
        for name in explicit_vars:
            var = self._vars.get(name)
            if var is None or var.fixed:
                continue

            current = var.input_value if var is not None else None
            current_scalar = safe_float(current)
            if current_scalar is None:
                continue

            used: list[tuple[object, float, float]] = []
            sum_dres2 = 0.0
            sum_dres_res = 0.0

            for rel, residual in rel_data:
                if name not in self._rels_to_vars.get(rel, ()):
                    continue
                dres = self._residual_derivative(rel, name, values, current=current_scalar)
                if dres is None:
                    continue
                used.append((rel, residual, dres))
                sum_dres2 += dres * dres
                sum_dres_res += dres * residual

            if not used or sum_dres2 == 0:
                continue

            dx = -sum_dres_res / sum_dres2
            target_scalar = safe_float(current_scalar + dx)
            if target_scalar is None:
                continue

            rel_tol = var.rel_tol or 0.0
            abs_tol = var.abs_tol or 0.0
            if within_tolerance(current_scalar, target_scalar, rel_tol=rel_tol, abs_tol=abs_tol):
                continue

            merged = {**values, name: target_scalar}
            if self._constraints_violated(merged, names=[name]):
                continue

            violated = False
            for rel, _, _ in used:
                if self._constraints_violated(merged, rel=rel):
                    violated = True
                    break
            if violated:
                continue

            if current_scalar != 0 and target_scalar != 0 and (current_scalar * target_scalar) > 0:
                change = abs(math.log10(abs(target_scalar / current_scalar)))
            else:
                scale = max(abs(current_scalar), abs(target_scalar), 1.0)
                change = abs(target_scalar - current_scalar) / scale

            next_values = dict(values)
            next_values[name] = target_scalar
            violations_after = len(self._violated_relations(next_values, rel_list))
            sse_after = 0.0
            for rel in rel_list:
                if any(n not in next_values for n in self._rels_to_vars.get(rel, ())):
                    continue
                residual = self._residual(rel, next_values, scaled=True)
                if residual is None:
                    continue
                sse_after += residual * residual

            key = (violations_after, sse_after, change)
            if self._candidate_better(key, best_key):
                meta = {
                    "sse_before": sse_before,
                    "dx": dx,
                    "relations": [rel.name for rel, _, _ in used],
                }
                meta.setdefault("sse_before", base_sse)
                meta.setdefault("violations_before", base_violations)
                meta["sse_after"] = sse_after
                meta["violations_after"] = violations_after
                best_key = key
                best_global = (name, change, target_scalar, "GLOBAL", meta, violations_after, sse_after)

        if best_global:
            name, change, target, rel_name, meta, violations_after, sse_after = best_global
            if violations_after < base_violations or (
                violations_after == base_violations and sse_after < base_sse
            ):
                rel_name = (meta.get("relations") if meta else None) or "GLOBAL"
                return ("global", name, change, target, rel_name, meta)
        return None

    def solve(self) -> None:
        """Solve the relation system in-place by iteratively enforcing relations and solving unknowns.
        
        Orchestrates the main solving loop:
        1. Initializes the solving process (early exit if check mode)
        2. Iterates passes until convergence or max passes reached
        3. Each pass: enforces relations, checks violations, solves unknowns
        4. Returns when converged or no progress can be made
        
        Args:
            none
            
        Returns:
            None
        """
        rel_nodes = self.relations
        rel_index = {rel: idx for idx, rel in enumerate(rel_nodes)}
        var_count = len(self._var_order)
        mode_overwrite = self.mode == "overwrite"
        self._log.extra["pass_id"] = self.pass_id if self.pass_id >= 0 else 0
        self._log.extra["mode"] = self.mode
        self._log.extra["inconsistency_warned"] = len(self._inconsistency_warned)
        self._log.info(
            "Starting solve: %s relations, %s variables, mode=%s, pass_mode=%s",
            len(rel_nodes),
            var_count,
            self.mode,
            mode_overwrite,
        )
        self._log.debug(
            "solve() called: n_relations=%s, n_variables=%s, mode=%s",
            len(rel_nodes),
            var_count,
            self.mode,
        )
        if self.mode == "check":
            self._log.debug("solve(): early exit for check mode")
            return  # Early exit for check mode

        # Seed the pending queue with all relations for the first pass.
        self._pending_rels.update(rel_nodes)

        passes = 0
        self._log.debug("solve(): entering main loop")
        while True:
            # Each pass enforces relations, updates violations, then solves blocks.
            self._log.debug(f"solve(): pass iteration {passes}")
            pass_label = self.pass_id if self.pass_id >= 0 else 0
            self._log.extra["pass_id"] = pass_label
            self._log.extra["mode"] = self.mode
            self._log.extra["inconsistency_warned"] = len(self._inconsistency_warned)
            self._log.info(
                "Pass %s: enforcing %s pending relations",
                pass_label,
                len(self._pending_rels),
            )
            self._log.debug(f"_execute_pass: pass_id={self.pass_id}, {len(self._pending_rels)} pending relations")

            # 1) Enforce forward/backward relations over the pending queue.
            _applied, touched = self._enforce_pending_relations(
                rel_index,
                mode_overwrite=mode_overwrite,
            )
            values = self._values_dict()
            violated_pending = self._violated_relations(values, list(touched))
            self._violated.difference_update(touched)
            self._violated.update(violated_pending)
            self._log.info("Pass %s: violated %s", pass_label, len(self._violated))
            self._log.debug(f"_execute_pass: {len(self._violated)} violations")
            self._pending_rels.clear()

            # 2) Group relations by unknown sets and try solving square blocks.
            unknown_map = self._build_unknown_map(rel_nodes, values)
            self._log.info(
                "0-unknown relations: %s",
                len(unknown_map.get(frozenset(), [])),
            )
            made_progress = self._solve_unknown_blocks(
                unknown_map,
                values,
                mode_overwrite=mode_overwrite,
            )
            if made_progress or self._pending_rels:
                # If we set values or queued new relations, keep iterating.
                continue

            # 3) No progress and no pending work: resolve violations or exit.
            if self._violated:
                if mode_overwrite and passes < self.max_passes:
                    values = self._values_dict()
                    violated_all = self._violated_relations(values)
                    if violated_all:
                        self._violated = set(violated_all)
                    result = self._select_culprit(set(violated_all), values, rel_nodes)
                    if result:
                        kind, name, change, target, rel_name, meta = result
                        is_global = kind == "global" and meta is not None
                        if is_global:
                            sse_before = meta.get("sse_before") if meta else None
                            sse_after = meta.get("sse_after") if meta else None
                            score = meta.get("score") if meta else None
                            self._log.info(
                                "Global override candidate: %s -> %s (delta %.3g, sse %s -> %s, score %s)",
                                name,
                                target,
                                change,
                                f"{sse_before:.3g}" if sse_before is not None else "n/a",
                                f"{sse_after:.3g}" if sse_after is not None else "n/a",
                                f"{score:.3g}" if score is not None else "n/a",
                            )
                        else:
                            self._log.info(
                                "Override candidate: %s -> %s (delta %.3g) from %s",
                                name,
                                target,
                                change,
                                rel_name,
                            )
                        suffix = " (global)" if is_global else ""
                        self._log.warning(
                            "Inconsistency: overriding %s by %.3g -> %s%s",
                            name,
                            change,
                            target,
                            suffix,
                        )
                        self.overrides[name] = {"value": target, "relation": rel_name}

                        if mode_overwrite:
                            self._start_pass()
                        else:
                            self._set_value(name, target, reason="override", relation=rel_name)
                        passes += 1
                        continue
                return

            return
    
    def _violated_relations(self, values: dict[str, object], rels: list | None = None) -> set[object]:
        """Return the set of violated relations for the given values."""
        rel_list = list(self.relations) if rels is None else rels
        return {rel for rel in rel_list if self._relation_status(rel, values)[0] == "VIOLATED"}

    def _relation_status(self, rel: object, values: dict[str, object]) -> tuple[str, float | None]:
        """Return (status, residual) for a relation given values."""
        if any(name not in values for name in self._rels_to_vars.get(rel, ())):
            return ("UNDECIDABLE", None)
        if self._constraints_violated(values, rel=rel):
            return ("VIOLATED", None)
        from .relation_util import check_relation_satisfied
        _, status, residual = check_relation_satisfied(rel, values, check_constraints=False)
        return status, residual


    def _culprit_for_relation(
        self,
        rel: object,
        values: dict[str, object],
    ) -> tuple[str, float, float] | None:
        """
        Identify which explicit variable needs adjustment to satisfy a violated relation.

        Returns:
            Tuple of (variable_name, relative_change, target_value) for the best culprit,
            or None if no suitable culprit found.
        """
        rel_vars = self._rels_to_vars.get(rel, ())
        if any(values.get(name) is None for name in rel_vars):
            self._log.debug("culprit_for_relation(%s): missing values for some variables", rel.name)
            return None

        explicit_vars = {
            name
            for name in rel_vars
            if (var := self._vars.get(name))
            and var.input_source == "explicit"
            and var.input_value is not None
        }

        self._log.debug("culprit_for_relation(%s): analyzing %s explicit variables", rel.name, len(explicit_vars))
        best: tuple[str, float, float] | None = None
        exp_scalar = self._expected(rel, values)
        if exp_scalar is None:
            return None
        act_scalar = safe_float(values.get(rel.output))
        if act_scalar is None:
            return None
        residual = act_scalar - exp_scalar
        rel_tol = rel.rel_tol_default or 0.0
        abs_tol = rel.abs_tol_default or 0.0
        rel_violated = not within_tolerance(act_scalar, exp_scalar, rel_tol=rel_tol, abs_tol=abs_tol)

        # Prefer the explicit output as culprit when it's inconsistent.
        if rel.output in explicit_vars and rel_violated:
            var = self._vars.get(rel.output)
            if var is not None and not var.fixed:
                current_scalar = safe_float(var.input_value)
                if current_scalar is not None:
                    target_scalar = exp_scalar
                    merged = {**values, rel.output: target_scalar}
                    if not self._constraints_violated(merged, rel=rel, names=[rel.output]):
                        if current_scalar and target_scalar and (current_scalar * target_scalar) > 0:
                            change = abs(math.log10(abs(target_scalar / current_scalar)))
                        else:
                            scale = max(abs(current_scalar), abs(target_scalar), 1.0)
                            change = abs(target_scalar - current_scalar) / scale
                        return (rel.output, change, target_scalar)

        for name in rel_vars:
            if name not in explicit_vars:
                continue

            var = self._vars.get(name)
            if var is None or var.fixed:
                continue

            current = safe_float(var.input_value)
            if current is None:
                continue

            target_scalar = None
            if name == rel.output:
                target_scalar = exp_scalar
            elif name in rel.inputs:
                dres = self._residual_derivative(rel, name, values, current=current)
                if dres is None:
                    continue
                target_scalar = current - residual / dres

            if target_scalar is None:
                continue

            merged = {**values, name: target_scalar}
            if self._constraints_violated(merged, rel=rel, names=[name]):
                continue

            rel_tol = var.rel_tol
            abs_tol = var.abs_tol
            if rel_tol is None and abs_tol is None:
                rel_tol = rel.rel_tol_default or 0.0
                abs_tol = rel.abs_tol_default or 0.0
            rel_tol = rel_tol or 0.0
            abs_tol = abs_tol or 0.0

            if within_tolerance(current, target_scalar, rel_tol=rel_tol, abs_tol=abs_tol):
                continue

            # Prefer an orders-of-magnitude change metric when both values are positive/non-zero.
            # This avoids treating huge multi-decade shifts as ~O(1) changes.
            if current and target_scalar and (current * target_scalar) > 0:
                change = abs(math.log10(abs(target_scalar / current)))
            else:
                scale = max(abs(current), abs(target_scalar), 1.0)
                change = abs(target_scalar - current) / scale
            if best is None or change < best[1]:
                best = (name, change, target_scalar)
                self._log.debug("  New best culprit: %s, change=%.6g, target=%.6g", name, change, target_scalar)

        if best:
            self._log.debug("culprit_for_relation(%s): best=%s, change=%.6g", rel.name, best[0], best[1])
        else:
            self._log.debug("culprit_for_relation(%s): no culprit found", rel.name)
        return best

    
    def diagnose_relations(
        self,
        values_override: dict[str, object] | None = None,
        *,
        return_culprits: bool = False,
    ) -> list[tuple[str, str, float | None]] | tuple[list[tuple[str, str, float | None]], dict[str, tuple[str, float, float]]]:
        """Evaluate relation consistency. Args: values_override. Returns: list of (name, status, residual)."""
        values = values_override or self._values_dict()
        results: list[tuple[str, str, float | None]] = []
        culprits: dict[str, tuple[str, float, float]] | None = {} if return_culprits else None
        for rel in self.relations:
            status, residual = self._relation_status(rel, values)
            results.append((rel.name, status, residual))
            if culprits is not None and status == "VIOLATED":
                if culprit := self._culprit_for_relation(rel, values):
                    culprits[rel.name] = culprit

        if culprits is not None:
            return results, culprits
        return results
    
    def diagnose_variables(self) -> list[tuple[str, str, int | None]]:
        """Diagnose variable consistency. Args: none. Returns: list of (name, status, rank)."""
        results: list[tuple[str, str, int | None]] = []
        for name in self._var_order:
            input_var = self._vars.get(name)
            if input_var is None:
                continue
            input_val = input_var.input_value
            computed_vals = input_var.values
            if input_val is None or not computed_vals:
                results.append((name, "UNDETERMINABLE", None))
                continue

            base = float(input_val)
            rel_tol = input_var.rel_tol or 0.0
            abs_tol = input_var.abs_tol or 0.0
            inconsistent = sum(
                1
                for value in computed_vals
                if value is not None
                and not within_tolerance(base, float(value), rel_tol=rel_tol, abs_tol=abs_tol)
            )
            results.append(
                (name, "INCONSISTENT", inconsistent)
                if inconsistent
                else (name, "CONSISTENT", None)
            )
        
        return results

    def export_relation_graph(self, path: str | Path = "relation_graph.html") -> Path:
        """Write an interactive HTML relation graph (variables as nodes, relations as edges)."""
        path = Path(path)
        values = self._values_dict()
        var_names: list[str] = []
        seen: set[str] = set()
        for name in self._var_order:
            if name not in seen:
                seen.add(name)
                var_names.append(name)
        for rel in self.relations:
            for name in getattr(rel, "variables", ()):
                if name is None or name in seen:
                    continue
                seen.add(name)
                var_names.append(name)

        nodes = []
        for name in var_names:
            title = name
            if name in values:
                title = f"{name}<br>value={values[name]}"
            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "title": title,
                    "shape": "dot",
                    "color": "#97c2fc",
                }
            )

        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        edges = []
        for idx, rel in enumerate(self.relations):
            color = palette[idx % len(palette)]
            output = getattr(rel, "output", None)
            if output is None:
                continue
            for name in getattr(rel, "inputs", ()):
                if name is None:
                    continue
                edges.append(
                    {
                        "from": name,
                        "to": output,
                        "label": getattr(rel, "name", ""),
                        "title": getattr(rel, "name", ""),
                        "relation": getattr(rel, "name", ""),
                        "arrows": "to",
                        "color": color,
                    }
                )

        html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Relation graph</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
      #mynetwork {{
        width: 100%;
        height: 800px;
        border: 1px solid #ddd;
        background: #fff;
      }}
    </style>
  </head>
  <body>
    <div id="mynetwork"></div>
    <script>
      const nodes = new vis.DataSet({json.dumps(nodes)});
      const edges = new vis.DataSet({json.dumps(edges)});
      const data = {{ nodes, edges }};
      const options = {{
        nodes: {{ shape: "dot", size: 18, font: {{ size: 16, face: "monospace" }} }},
        edges: {{ arrows: {{ to: {{ enabled: true }} }}, font: {{ size: 12, align: "middle" }} }},
        interaction: {{ hover: true }},
        physics: {{ barnesHut: {{ springLength: 140, springConstant: 0.03 }} }}
      }};
      const container = document.getElementById("mynetwork");
      new vis.Network(container, data, options);
    </script>
  </body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path
    
