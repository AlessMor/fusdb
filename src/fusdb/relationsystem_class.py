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
import numpy as np
import sympy as sp

from .registry import (
    allowed_variable_constraints,
    allowed_variable_soft_constraints,
    allowed_variable_ndim,
)
from .variable_class import Variable
from .variable_util import make_variable
from .utils import (
    as_profile_array,
    compare_plasma_volume_with_integrated_dv,
    safe_float,
    within_tolerance,
    normalize_solver_mode,
    ensure_list,
    normalize_tags_to_tuple,
)

logger = logging.getLogger(__name__)


def make_logger(
    module_logger: logging.Logger,
    owner: str,
    *,
    verbose: bool,
) -> logging.Logger:
    """Return a child logger configured for the requested verbosity."""
    log = module_logger.getChild(owner)
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    return log


def log_message(log: logging.Logger, level: int, msg: str, *args: object) -> None:
    """Emit one log message preserving the class call-site location."""
    log.log(level, msg, *args, stacklevel=2)

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
    enforce_constraint_tags: tuple[str, ...] = field(default_factory=tuple)
    enforce_constraint_names: tuple[str, ...] = field(default_factory=tuple)
    _log: logging.Logger = field(init=False, repr=False)
    # Single graph object for topology and variable storage.
    # Keys:
    # - vars: {name -> Variable | None}
    # - var_order: [name, ...] (stable iteration order)
    # - vars_to_rels / rels_to_vars: adjacency maps
    _graph: dict[str, object] = field(init=False, default_factory=dict, repr=False)
    # Constraint object for all hard/soft and compiled constraints.
    # Keys:
    # - var_hard / var_soft
    # - rel_hard / rel_soft
    # - var_bounds
    # - compiled
    _constraints: dict[str, object] = field(init=False, default_factory=dict, repr=False)
    # Runtime state container to keep private attributes minimal.
    # Keys:
    # - pass_id, overrides, pending_rels, violated
    # - inconsistency_warned, soft_constraint_warned, volume_consistency_warned
    # - enforced_relations, enforced_var_names
    # - eval_scalar_order, eval_var_index, eval_plan
    _state: dict[str, object] = field(init=False, default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize maps and caches. Args: none. Returns: None."""
        # Per-instance logger.
        self._log = make_logger(
            logger,
            self.__class__.__name__,
            verbose=self.verbose,
        )

        # Step 0: initialize compact runtime containers.
        self._state = {
            "pass_id": -1,
            "overrides": {},
            "pending_rels": set(),
            "violated": set(),
            "inconsistency_warned": set(),
            "soft_constraint_warned": set(),
            "volume_consistency_warned": set(),
            "enforced_relations": set(),
            "enforced_var_names": set(),
            "eval_scalar_order": [],
            "eval_var_index": {},
            "eval_plan": [],
        }

        # Step 1: normalize constructor payloads into concrete relation/variable lists.
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
        log_message(
            self._log,
            logging.DEBUG,
            "RelationSystem.__post_init__: mode=%s, n_relations=%s, n_variables=%s",
            self.mode,
            len(rel_nodes),
            len(provided_vars),
        )
        # Normalize mode and set flags
        self.mode = normalize_solver_mode(self.mode)
        log_message(self._log, logging.DEBUG, f"Normalized mode: {self.mode}")

        # Step 2: initialize graph + constraint containers.
        self._graph = {
            "vars": {},
            "var_order": [],
            "vars_to_rels": {},
            "rels_to_vars": {},
        }
        self._constraints = {
            "var_hard": {},
            "var_soft": {},
            "rel_hard": {},
            "rel_soft": {},
            "var_bounds": {},
            "compiled": {},
        }
        
        # Step 3: build relation<->variable connectivity first.
        seen_names: set[str] = set()
        duplicate_name_indices: dict[str, list[int]] = {}
        var_by_name = {name: var for name, var in provided_pairs}
        self._graph["var_order"] = [name for name, _ in provided_pairs]
        var_names: set[str] = set(self._graph["var_order"])

        self._graph["rels_to_vars"] = {}
        self._graph["vars_to_rels"] = {}
        for idx, rel in enumerate(rel_nodes):
            rel_name = getattr(rel, "name", None) or f"relation_{idx}"
            if rel_name in seen_names:
                duplicate_name_indices.setdefault(rel_name, []).append(idx)
            else:
                seen_names.add(rel_name)

            vars_for_rel = tuple(v for v in getattr(rel, "variables", {}) if v is not None)
            self._graph["rels_to_vars"][rel] = vars_for_rel
            for name in vars_for_rel:
                self._graph["vars_to_rels"].setdefault(name, set()).add(rel)
            var_names.update(vars_for_rel)

        if duplicate_name_indices:
            duplicate_names = sorted(duplicate_name_indices)
            sample = ", ".join(duplicate_names[:8])
            if len(duplicate_names) > 8:
                sample = f"{sample}, ..."
            duplicate_total = sum(len(indices) for indices in duplicate_name_indices.values())
            log_message(
                self._log,
                logging.INFO,
                "Detected %d duplicate relation entries across %d names; names are not unique keys. Examples: %s",
                duplicate_total,
                len(duplicate_names),
                sample,
            )

        # Ensure all provided variables appear in the adjacency map (even if isolated).
        for name in self._graph["var_order"]:
            self._graph["vars_to_rels"].setdefault(name, set())

        var_names_sorted = sorted(var_names)
        for name in var_names_sorted:
            if name not in self._graph["var_order"]:
                self._graph["var_order"].append(name)
        log_message(self._log, logging.DEBUG, f"Total variables in system: {len(var_names_sorted)}")

        # Step 4: attach per-variable/per-relation constraints metadata.
        var_with_constraints = 0
        var_with_soft_constraints = 0
        for name in var_names_sorted:
            var = var_by_name.get(name)
            if var is not None and var.rel_tol is None and var.abs_tol is None:
                var.rel_tol = self.default_rel_tol
            cons = allowed_variable_constraints(name)
            soft_cons = allowed_variable_soft_constraints(name)
            if cons:
                var_with_constraints += 1
            if soft_cons:
                var_with_soft_constraints += 1
            self._graph["vars"][name] = var
            self._constraints["var_hard"][name] = cons
            self._constraints["var_soft"][name] = soft_cons

        rel_with_constraints = 0
        rel_with_soft_constraints = 0
        for rel in rel_nodes:
            cons = tuple(getattr(rel, "constraints", ()) or ())
            soft_cons = tuple(getattr(rel, "soft_constraints", ()) or ())
            if cons:
                rel_with_constraints += 1
            if soft_cons:
                rel_with_soft_constraints += 1
            self._constraints["rel_hard"][rel] = cons
            self._constraints["rel_soft"][rel] = soft_cons

        # Step 5: pre-compile all constraints and infer static variable bounds once.
        all_constraints: set[str] = set()
        for cons in itertools.chain(
            self._constraints["var_hard"].values(),
            self._constraints["var_soft"].values(),
            self._constraints["rel_hard"].values(),
            self._constraints["rel_soft"].values(),
        ):
            all_constraints.update(cons)
        for constraint in all_constraints:
            self._compile_constraint(constraint)

        self._constraints["var_bounds"] = {name: self._compute_var_bounds(name) for name in self._graph["var_order"]}

        log_message(
            self._log,
            logging.DEBUG,
            "Built constraints: %s variables with constraints, %s relations with constraints",
            var_with_constraints,
            rel_with_constraints,
        )
        if var_with_soft_constraints or rel_with_soft_constraints:
            log_message(
                self._log,
                logging.DEBUG,
                "Built soft constraints: %s variables with soft constraints, %s relations with soft constraints",
                var_with_soft_constraints,
                rel_with_soft_constraints,
            )

        # Step 6: mark enforced constraints and initialize solver runtime queues.
        enforce_tags = set(normalize_tags_to_tuple(self.enforce_constraint_tags))
        enforce_names = {str(name) for name in (self.enforce_constraint_names or ())}
        self._state["enforced_var_names"] = set(enforce_names)
        if enforce_tags or enforce_names:
            for rel in rel_nodes:
                target_name = rel.preferred_target
                rel_name = getattr(rel, "name", target_name)
                if rel_name in enforce_names or (target_name is not None and target_name in enforce_names):
                    self._state["enforced_relations"].add(rel)
                    continue
                if enforce_tags and any(tag in enforce_tags for tag in (getattr(rel, "tags", ()) or ())):
                    self._state["enforced_relations"].add(rel)
        
        self._state["violated"]: set[object] = set()
        self._state["pending_rels"]: set[object] = set(rel_nodes)
        self._build_eval_plan()
        if self.mode == 'overwrite':
            self._start_pass()

    @staticmethod
    def _all_tolerances(
        left: object,
        right: object,
        *,
        rel_tol: float,
        abs_tol: float,
    ) -> bool:
        """Return whether values satisfy all active tolerance checks."""
        try:
            left_arr = np.asarray(left, dtype=float)
            right_arr = np.asarray(right, dtype=float)
        except Exception:
            return False

        if left_arr.shape != right_arr.shape:
            return False
        diff = np.abs(left_arr - right_arr)
        use_rel = rel_tol > 0.0
        use_abs = abs_tol > 0.0
        if not use_rel and not use_abs:
            return bool(np.all(diff == 0.0))
        if use_abs and not np.all(diff <= abs_tol):
            return False
        if use_rel:
            scale = np.maximum(np.maximum(np.abs(left_arr), np.abs(right_arr)), 1.0)
            if not np.all(diff <= rel_tol * scale):
                return False
        return True

    def _values_equal_for_var(
        self,
        var: Variable,
        left: object,
        right: object,
    ) -> bool:
        """Return whether two values should be treated as unchanged for a variable."""
        rel_tol = var.rel_tol if var.rel_tol is not None else self.default_rel_tol
        abs_tol = var.abs_tol if var.abs_tol is not None else 0.0
        if left is None or right is None:
            return left is right

        if var.ndim == 1:
            y_left = as_profile_array(left)
            y_right = as_profile_array(right)
            if y_left is None or y_right is None:
                return False
            return bool(
                self._all_tolerances(y_left, y_right, rel_tol=rel_tol, abs_tol=abs_tol)
            )

        lv = safe_float(left)
        rv = safe_float(right)
        if lv is not None and rv is not None:
            return self._all_tolerances(lv, rv, rel_tol=rel_tol, abs_tol=abs_tol)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return self._all_tolerances(left, right, rel_tol=rel_tol, abs_tol=abs_tol)

        if isinstance(left, sp.Basic) or isinstance(right, sp.Basic):
            try:
                if sp.simplify(left - right) == 0:
                    return True
            except Exception:
                pass
            lv = safe_float(left.evalf() if isinstance(left, sp.Basic) else left)
            rv = safe_float(right.evalf() if isinstance(right, sp.Basic) else right)
            if lv is not None and rv is not None:
                return self._all_tolerances(lv, rv, rel_tol=rel_tol, abs_tol=abs_tol)

        try:
            return bool(left == right)
        except Exception:
            return False

    def _build_eval_plan(self) -> None:
        """Build scalar variable index and relation evaluation plan."""
        var_order = self._graph["var_order"]
        scalar_names: list[str] = []
        for name in var_order:
            if self._var_ndim(name) == 1:
                continue
            scalar_names.append(name)

        self._state["eval_scalar_order"] = scalar_names
        self._state["eval_var_index"] = {name: idx for idx, name in enumerate(scalar_names)}
        self._state["eval_plan"] = []
        for rel in self.relations:
            target_name = rel.preferred_target
            input_names = tuple(rel.required_inputs(target_name))
            output_idx = self._state["eval_var_index"].get(target_name) if target_name is not None else None
            input_idx = tuple(self._state["eval_var_index"].get(name) for name in input_names)
            self._state["eval_plan"].append((rel, output_idx, input_names, input_idx))

    def _var_ndim(self, name: str) -> int:
        """Return variable ndim from object metadata, with registry fallback."""
        var = self._graph["vars"].get(name)
        if var is not None:
            return int(var.ndim)
        return int(allowed_variable_ndim(name))

    def _to_scalar_values(self, values: dict[str, object]) -> dict[str, object]:
        """Return values where profile payloads are reduced to profile means."""
        values_scalar: dict[str, object] = {}
        for name, value in values.items():
            if self._var_ndim(name) == 1:
                arr = as_profile_array(value)
                mean = float(np.mean(arr)) if arr is not None else None
                if mean is None:
                    mean = safe_float(value)
                values_scalar[name] = mean
            else:
                values_scalar[name] = value
        return values_scalar

    @property
    def variables_dict(self) -> dict[str, Variable]:
        """Return current variables keyed by name (dict graph is source of truth)."""
        vars_map: dict[str, Variable] = {}
        for name in self._graph["var_order"]:
            var = self._graph["vars"].get(name)
            if var is not None:
                vars_map[name] = var
        return vars_map
    
    def _get_value(self, name: str) -> object | None:
        """Get the effective value for a variable.

        In overwrite mode this returns the current value (or falls back to the
        input value). In check mode this returns current or input value.

        Args:
            name: Variable name to retrieve

        Returns:
            The effective value or None if not available
        """
        var = self._graph["vars"].get(name)
        if var is None:
            return None

        current = var.current_value
        if self.mode == 'overwrite':
            if current is not None:
                return current
            input_val = var.input_value
            if input_val is None:
                log_message(self._log, logging.DEBUG, f"_get_value({name}): no current value, no input value")
            return input_val

        if current is not None:
            return current

        input_val = var.input_value
        if input_val is None:
            log_message(self._log, logging.DEBUG, f"_get_value({name}): no current value, no input value (non-overwrite mode)")
        return input_val

    def _values_dict(self) -> dict[str, object]:
        """Return mapping of effective values."""
        return {n: v for n in self._graph["var_order"] if (v := self._get_value(n)) is not None}

    def _compile_constraint(self, constraint: str) -> tuple[tuple[str, ...], object | None]:
        cached = self._constraints["compiled"].get(constraint)
        if cached is not None:
            return cached
        try:
            expr = sp.sympify(constraint)
        except Exception:
            cached = ((), None)
            self._constraints["compiled"][constraint] = cached
            return cached
        if not hasattr(expr, "free_symbols"):
            try:
                cached = ((), bool(expr))
            except Exception:
                cached = ((), None)
            self._constraints["compiled"][constraint] = cached
            return cached
        symbols = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
        if not symbols:
            try:
                cached = ((), bool(expr))
            except Exception:
                cached = ((), None)
            self._constraints["compiled"][constraint] = cached
            return cached
        names = tuple(sym.name for sym in symbols)
        try:
            func = sp.lambdify(symbols, expr, modules=["numpy"])
        except Exception:
            func = None
        cached = (names, func)
        self._constraints["compiled"][constraint] = cached
        return cached

    def _constraint_result(
        self,
        constraint: str,
        values_scalar: dict[str, object],
    ) -> bool | None:
        cached = self._constraints["compiled"].get(constraint)
        if cached is None:
            return None
        names, func = cached
        if func is None:
            return None
        if not names:
            try:
                return bool(func)
            except Exception:
                return None
        args: list[object] = []
        for name in names:
            val = values_scalar.get(name)
            if val is None:
                return None
            args.append(val)
        try:
            return bool(func(*args))
        except Exception:
            return None

    def _constraint_violations(
        self,
        constraints: Iterable[str],
        values_scalar: dict[str, object],
    ) -> list[str]:
        return [
            constraint
            for constraint in constraints
            if self._constraint_result(constraint, values_scalar) is False
        ]

    def _warn_if_volume_integral_mismatch(
        self,
        values: dict[str, object],
    ) -> None:
        """Warn when geometry-based integral(dV) is inconsistent with ``V_p``."""
        V_p = safe_float(values.get("V_p"))
        R = safe_float(values.get("R"))
        a = safe_float(values.get("a"))
        kappa = safe_float(values.get("kappa_95")) or safe_float(values.get("kappa"))
        if V_p is None or R is None or a is None or kappa is None:
            return

        key = f"{V_p:.6g}:{R:.6g}:{a:.6g}:{kappa:.6g}"
        if key in self._state["volume_consistency_warned"]:
            return

        ok, V_int, V_ref = compare_plasma_volume_with_integrated_dv(
            V_p=V_p,
            R=R,
            a=a,
            kappa=kappa,
            rel_tol=0.01,
            abs_tol=0.0,
            warn=False,
        )
        if ok:
            return
        self._state["volume_consistency_warned"].add(key)
        if V_int is None or V_ref is None:
            return
        delta = abs(V_int - V_ref) / max(abs(V_ref), 1.0)
        log_message(self._log, logging.WARNING, 
            "Volume consistency warning: integral(dV)=%.6g differs from V_p=%.6g (rel_delta=%.3f%%, tol=1%%).",
            V_int,
            V_ref,
            100.0 * delta,
        )

    def _accept_candidate_values(
        self,
        solved: dict[str, object],
        *,
        rels: list[object],
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
            rels_to_check = self._graph["vars_to_rels"].get(name, ())
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
                    log_message(self._log, logging.DEBUG, 
                        "    ✗ Candidate for %s increases violations",
                        name,
                    )
                    return False

        accepted: dict[str, object] = {}
        for name, value in solved.items():
            var = self._graph["vars"].get(name)
            if var is not None and var.fixed:
                return False

            is_profile = self._var_ndim(name) == 1
            if is_profile:
                accepted[name] = value
                continue

            value_scalar = safe_float(value)
            if value_scalar is None:
                return False

            cur_value = base_values.get(name)
            cur_scalar = safe_float(cur_value) if cur_value is not None else None
            rel_tol = var.rel_tol if var and var.rel_tol is not None else self.default_rel_tol
            abs_tol = var.abs_tol if var and var.abs_tol is not None else 0.0

            if warn_input and cur_scalar is not None:
                input_value = var.input_value if var is not None else None
                if input_value is not None and name not in self._state["inconsistency_warned"]:
                    input_scalar = safe_float(input_value)
                    if input_scalar is not None and not within_tolerance(
                        input_scalar, value_scalar, rel_tol=rel_tol, abs_tol=abs_tol
                    ):
                        rel_name = (
                            relation
                            if isinstance(relation, str)
                            else (relation[0] if relation else "unknown")
                        )
                        log_message(self._log, logging.WARNING, 
                            "Inconsistency: relation '%s' computed %s = %.3g, but input specifies %s = %.3g",
                            rel_name,
                            name,
                            value_scalar,
                            name,
                            input_scalar,
                        )
                        self._state["inconsistency_warned"].add(name)

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

            if protect_explicit and name in self._state["overrides"] and cur_scalar is not None:
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
                reason=reason,
                relation=relation,
            )
        return True

    
    def _set_value(
        self,
        name: str,
        value: object,
        *,
        reason: str | None = None,
        relation: str | list[str] | None = None,
        force: bool = False,
    ) -> None:
        """Store a solved value in values map and update caches/pending relations."""
        log_message(self._log, logging.DEBUG, f"_set_value({name}={value}, pass_mode={self.mode == 'overwrite'}, reason={reason})")
        var = self._graph["vars"].get(name)
        if var is None:
            log_message(self._log, logging.DEBUG, f"Creating new variable: {name}")
            var = make_variable(name=name, ndim=self._var_ndim(name))
            if var.rel_tol is None and var.abs_tol is None:
                var.rel_tol = self.default_rel_tol
            if name not in self._graph["var_order"]:
                self._graph["var_order"].append(name)
            self._graph["vars"][name] = var
            self._constraints["var_hard"].setdefault(name, allowed_variable_constraints(name))
            self._constraints["var_soft"].setdefault(name, allowed_variable_soft_constraints(name))
            self._graph["vars_to_rels"].setdefault(name, set())
            self._constraints["var_bounds"][name] = self._compute_var_bounds(name)
            self._build_eval_plan()
        if not force and var.current_value is not None and self._values_equal_for_var(var, var.current_value, value):
            log_message(self._log, logging.DEBUG, f"  No change for {name}")
            return

        changed = var.add_value(
            value,
            pass_id=self._state["pass_id"],
            reason=reason,
            as_input=reason in ("input", "default"),
        )
        if not changed:
            log_message(self._log, logging.DEBUG, f"  No change for {name}")
            return
        rels: set[object] = set(self._graph["vars_to_rels"].get(name, ()))
        if rels:
            log_message(self._log, logging.DEBUG, f"  {name} updated, adding {len(rels)} relations to pending")
            self._state["pending_rels"].update(rels)
    
    def _residual(self, rel: object, values: dict[str, object], *, scaled: bool = False) -> float | None:
        values_scalar = self._to_scalar_values(values)
        try:
            expected = rel.evaluate(values)
        except Exception:
            return None
        expected_scalar = safe_float(expected)
        if expected_scalar is None:
            return None
        target_name = rel.preferred_target
        if target_name is None:
            return None
        actual_scalar = safe_float(values_scalar.get(target_name))
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
        values_scalar = self._to_scalar_values(values)
        if name == rel.preferred_target:
            return 1.0
        if self._var_ndim(name) == 1:
            return None
        if current is None:
            current = safe_float(values_scalar.get(name))
        if current is None:
            return None
        step = 1e-6 * max(abs(current), 1.0)
        v_plus = dict(values)
        v_minus = dict(values)
        v_plus[name] = current + step
        v_minus[name] = current - step
        try:
            f_plus = safe_float(rel.evaluate(v_plus))
        except Exception:
            f_plus = None
        try:
            f_minus = safe_float(rel.evaluate(v_minus))
        except Exception:
            f_minus = None
        if f_plus is None or f_minus is None:
            return None
        dfdx = (f_plus - f_minus) / (2 * step)
        if dfdx == 0 or not math.isfinite(dfdx):
            return None
        return -dfdx

    def _constraints_violated(
        self,
        values: dict[str, object],
        *,
        rel: object | None = None,
        names: Iterable[str] | None = None,
        include_soft: bool = False,
    ) -> bool:
        values_scalar = self._to_scalar_values(values)
        soft_for_rel = include_soft or (rel in self._state["enforced_relations"] if rel is not None else False)
        if rel is not None:
            if self._constraint_violations(self._constraints["rel_hard"].get(rel, ()), values_scalar):
                return True
            if soft_for_rel and self._constraint_violations(
                self._constraints["rel_soft"].get(rel, ()), values_scalar
            ):
                return True
        if names is None:
            names = self._graph["rels_to_vars"].get(rel, ()) if rel is not None else ()
        for name in names:
            if self._constraint_violations(self._constraints["var_hard"].get(name, ()), values_scalar):
                return True
            soft_for_name = include_soft or (name in self._state["enforced_var_names"])
            if soft_for_name and self._constraint_violations(
                self._constraints["var_soft"].get(name, ()), values_scalar
            ):
                return True
        return False

    def _warn_soft_constraints(self, values: dict[str, object]) -> None:
        """Emit warnings for soft-constraint violations (once per constraint)."""
        values_scalar = self._to_scalar_values(values)

        for name in self._graph["var_order"]:
            if name in self._state["enforced_var_names"]:
                continue
            cons = self._constraints["var_soft"].get(name, ())
            for constraint in self._constraint_violations(cons, values_scalar):
                key = f"variable:{name}:{constraint}"
                if key in self._state["soft_constraint_warned"]:
                    continue
                self._state["soft_constraint_warned"].add(key)
                log_message(self._log, logging.WARNING, "Soft constraint violated (%s %s): %s", "variable", name, constraint)

        for rel in self.relations:
            if rel in self._state["enforced_relations"]:
                continue
            rel_name = getattr(
                rel,
                "name",
                rel.preferred_target,
            )
            cons = self._constraints["rel_soft"].get(rel, ())
            for constraint in self._constraint_violations(cons, values_scalar):
                key = f"relation:{rel_name}:{constraint}"
                if key in self._state["soft_constraint_warned"]:
                    continue
                self._state["soft_constraint_warned"].add(key)
                log_message(self._log, logging.WARNING, "Soft constraint violated (%s %s): %s", "relation", rel_name, constraint)

    @staticmethod
    def _brent_root(
        f,
        a: float,
        b: float,
        fa: float,
        fb: float,
        *,
        abs_tol: float,
        rel_tol: float,
        max_iter: int,
    ) -> float | None:
        """Return root in [a, b] using a Brent-style method."""
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(fa) and math.isfinite(fb)):
            return None
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            return None

        c, fc = a, fa
        d = e = b - a

        for _ in range(max_iter):
            if fb * fc > 0.0:
                c, fc = a, fa
                d = e = b - a

            if abs(fc) < abs(fb):
                a, b, c = b, c, b
                fa, fb, fc = fb, fc, fb

            tol = abs_tol + rel_tol * max(abs(b), 1.0)
            m = 0.5 * (c - b)
            if abs(m) <= tol or fb == 0.0:
                return b

            if abs(e) >= tol and abs(fa) > abs(fb):
                s = fb / fa
                if a == c:
                    p = 2.0 * m * s
                    q = 1.0 - s
                else:
                    q = fa / fc
                    r = fb / fc
                    p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)
                if p > 0.0:
                    q = -q
                p = abs(p)
                if q != 0.0 and 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                    e = d
                    d = p / q
                else:
                    d = m
                    e = m
            else:
                d = m
                e = m

            a, fa = b, fb
            if abs(d) > tol:
                b += d
            else:
                b += tol if m > 0.0 else -tol

            fb = f(b)
            if fb is None or not math.isfinite(fb):
                return None

        return None

    def _numeric_inverse_single_scalar(
        self,
        rel: object,
        unknown: str,
        values_map: dict[str, object],
    ) -> float | None:
        """Solve one scalar unknown by root-finding relation residual."""
        if self._var_ndim(unknown) == 1:
            return None
        target_name = rel.preferred_target
        if target_name is None:
            return None
        if unknown == target_name:
            return None
        if target_name not in values_map:
            return None

        target = safe_float(values_map.get(target_name))
        if target is None:
            return None
        input_names = rel.required_inputs(target_name)
        if any(name != unknown and values_map.get(name) is None for name in input_names):
            return None

        out_var = self._graph["vars"].get(target_name)
        out_rel_tol = rel.rel_tol_default
        out_abs_tol = rel.abs_tol_default
        if out_rel_tol is None:
            out_rel_tol = out_var.rel_tol if out_var and out_var.rel_tol is not None else self.default_rel_tol
        if out_abs_tol is None:
            out_abs_tol = out_var.abs_tol if out_var and out_var.abs_tol is not None else 0.0
        out_rel_tol = 0.0 if out_rel_tol is None else float(out_rel_tol)
        out_abs_tol = 0.0 if out_abs_tol is None else float(out_abs_tol)

        unknown_var = self._graph["vars"].get(unknown)
        x_rel_tol = (
            unknown_var.rel_tol
            if unknown_var is not None and unknown_var.rel_tol is not None
            else self.default_rel_tol
        )
        x_abs_tol = (
            unknown_var.abs_tol
            if unknown_var is not None and unknown_var.abs_tol is not None
            else 0.0
        )

        def _residual(x: float) -> float | None:
            merged = dict(values_map)
            merged[unknown] = x
            try:
                expected = rel.evaluate(merged)
            except Exception:
                return None
            expected_scalar = safe_float(expected)
            if expected_scalar is None:
                return None
            return target - expected_scalar

        def _is_solution(x: float, fx: float | None = None) -> bool:
            merged = dict(values_map)
            merged[unknown] = x
            try:
                expected = rel.evaluate(merged)
            except Exception:
                return False
            expected_scalar = safe_float(expected)
            if expected_scalar is None:
                return False
            return within_tolerance(target, expected_scalar, rel_tol=out_rel_tol, abs_tol=out_abs_tol)

        lower, upper = self._constraints["var_bounds"].get(unknown, (None, None))
        if lower is not None and upper is not None and lower >= upper:
            return None

        guess_candidates: list[float] = []
        current = safe_float(values_map.get(unknown))
        if current is not None:
            guess_candidates.append(current)
        if unknown_var is not None:
            input_guess = safe_float(unknown_var.input_value)
            if input_guess is not None:
                guess_candidates.append(input_guess)
        if rel.initial_guesses and unknown in rel.initial_guesses:
            try:
                rel_guess = safe_float(rel.initial_guesses[unknown](values_map))
            except Exception:
                rel_guess = None
            if rel_guess is not None:
                guess_candidates.append(rel_guess)
        if lower is not None and upper is not None and math.isfinite(lower) and math.isfinite(upper):
            guess_candidates.append(0.5 * (lower + upper))
        guess_candidates.extend([1.0, 0.0, -1.0])

        def _clip(x: float) -> float:
            if lower is not None:
                x = max(x, lower)
            if upper is not None:
                x = min(x, upper)
            return x

        bracket: tuple[float, float, float, float] | None = None

        if lower is not None and upper is not None and math.isfinite(lower) and math.isfinite(upper):
            fa = _residual(lower)
            fb = _residual(upper)
            if fa is not None and fb is not None:
                if _is_solution(lower, fa):
                    return lower
                if _is_solution(upper, fb):
                    return upper
                if fa * fb < 0.0:
                    bracket = (lower, upper, fa, fb)
            if bracket is None:
                n_scan = 41
                xs = np.linspace(lower, upper, n_scan, dtype=float)
                prev_x = float(xs[0])
                prev_f = _residual(prev_x)
                if prev_f is not None and _is_solution(prev_x, prev_f):
                    return prev_x
                for x in xs[1:]:
                    x_val = float(x)
                    f_val = _residual(x_val)
                    if f_val is None:
                        prev_x, prev_f = x_val, f_val
                        continue
                    if _is_solution(x_val, f_val):
                        return x_val
                    if prev_f is not None and prev_f * f_val < 0.0:
                        bracket = (prev_x, x_val, prev_f, f_val)
                        break
                    prev_x, prev_f = x_val, f_val

        if bracket is None:
            span_factors = (1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4)
            for guess in guess_candidates:
                if guess is None or not math.isfinite(guess):
                    continue
                center = _clip(float(guess))
                for factor in span_factors:
                    span = factor * max(abs(center), 1.0)
                    a = _clip(center - span)
                    b = _clip(center + span)
                    if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
                        continue
                    fa = _residual(a)
                    fb = _residual(b)
                    if fa is None or fb is None:
                        continue
                    if _is_solution(a, fa):
                        return a
                    if _is_solution(b, fb):
                        return b
                    if fa * fb < 0.0:
                        bracket = (a, b, fa, fb)
                        break
                if bracket is not None:
                    break

        if bracket is None:
            return None

        a, b, fa, fb = bracket
        root = self._brent_root(
            _residual,
            a,
            b,
            fa,
            fb,
            abs_tol=0.0 if x_abs_tol is None else float(x_abs_tol),
            rel_tol=0.0 if x_rel_tol is None else float(x_rel_tol),
            max_iter=100,
        )
        if root is None:
            return None
        return root if _is_solution(root) else None


    def _solve_for_value(
        self,
        rel: object,
        name: str,
        values_map: dict[str, object],
        *,
        prefer_eval_output: bool = False,
    ) -> object | None:
        target_name = rel.preferred_target
        input_names = rel.required_inputs(target_name)
        if (
            prefer_eval_output
            and target_name is not None
            and name == target_name
            and all(n in values_map for n in input_names)
        ):
            try:
                return rel.evaluate(values_map)
            except Exception:
                return None

        has_profile_input = any(
            self._var_ndim(input_name) == 1
            for input_name in input_names
        )
        solved = None
        # Avoid profile-mean back-solving: profile-dependent inversions must be explicit
        # or handled by numeric fallback on the real relation evaluation.
        if not (has_profile_input and (target_name is None or name != target_name)):
            values_scalar = self._to_scalar_values(values_map)
            try:
                solved = rel.solve_for_value(name, values_scalar)
            except Exception:
                solved = None

        if solved is None and has_profile_input and (target_name is None or name != target_name):
            solved = self._numeric_inverse_single_scalar(rel, name, values_map)

        if solved is None and target_name is not None and name == target_name:
            try:
                solved = rel.evaluate(values_map)
            except Exception:
                return None
        return solved


    def _compute_var_bounds(self, name: str) -> tuple[float | None, float | None]:
        lower: float | None = None
        upper: float | None = None

        for constraint in self._constraints["var_hard"].get(name, ()):
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
        values_scalar = self._to_scalar_values(values)
        residuals: list[float] = []
        for constraint in constraints:
            ok = self._constraint_result(constraint, values_scalar)
            if ok is None:
                residuals.append(penalty)
                continue
            residuals.append(0.0 if ok else penalty)
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
                res += self._constraint_residuals(self._constraints["var_hard"].get(name, ()), merged, penalty=penalty)
            for rel in relations:
                res += self._constraint_residuals(self._constraints["rel_hard"].get(rel, ()), merged, penalty=penalty)
            return res

        bounds = [self._constraints["var_bounds"].get(name, (None, None)) for name in unknowns]
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
            rel_vars = set(rel.variables)
            if rel_vars.issubset(set(unknowns)):
                return None

        if len(unknowns) == 1 and len(relations) == 1:
            rel = relations[0]
            unknown = unknowns[0]
            solved = self._solve_for_value(rel, unknown, values_map, prefer_eval_output=True)
            return None if solved is None else {unknown: solved}

        return self._least_squares_block_compact(relations, unknowns, values_map)

    def _start_pass(self) -> None:
        """Start a new override pass and seed inputs. Args: none. Returns: None."""
        self._state["pass_id"] += 1
        self._state["pending_rels"] = set(self.relations)
        
        for name in self._graph["var_order"]:
            var = self._graph["vars"].get(name)
            if var is None or var.input_source is None:
                continue
            
            override = self._state["overrides"].get(name)
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

        base_violations = len(self._violated_relations(values))
        base_sse = 0.0
        for rel in rel_nodes:
            if any(name not in values for name in self._graph["rels_to_vars"].get(rel, ())):
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
                for var in self._graph["rels_to_vars"].get(rel, ()):
                    for rel_neighbor in self._graph["vars_to_rels"].get(var, ()):
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
                var = self._graph["vars"].get(name)
                override = self._state["overrides"].get(name)
                current = override.get("value") if override else (var.input_value if var is not None else None)
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
                    if any(n not in next_values for n in self._graph["rels_to_vars"].get(rel, ())):
                        continue
                    residual = self._residual(rel, next_values, scaled=True)
                    if residual is None:
                        continue
                    sse_after += residual * residual
                key = (violations_after, sse_after, change)
                if _candidate_better(key, best_key):
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
            for name in self._graph["var_order"]
            if (var := self._graph["vars"].get(name)) and var.input_source == "explicit" and var.input_value is not None
        }

        rel_data: list[tuple[object, float]] = []
        for rel in rel_list:
            if any(name not in values or values.get(name) is None for name in self._graph["rels_to_vars"].get(rel, ())):
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
            var = self._graph["vars"].get(name)
            if var is None or var.fixed:
                continue

            override = self._state["overrides"].get(name)
            current = override.get("value") if override else (var.input_value if var is not None else None)
            current_scalar = safe_float(current)
            if current_scalar is None:
                continue

            used: list[tuple[object, float, float]] = []
            sum_dres2 = 0.0
            sum_dres_res = 0.0

            for rel, residual in rel_data:
                if name not in self._graph["rels_to_vars"].get(rel, ()):
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
                if any(n not in next_values for n in self._graph["rels_to_vars"].get(rel, ())):
                    continue
                residual = self._residual(rel, next_values, scaled=True)
                if residual is None:
                    continue
                sse_after += residual * residual

            key = (violations_after, sse_after, change)
            if _candidate_better(key, best_key):
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

    def _resolve_coupled_violations(
        self,
        values: dict[str, object],
    ) -> bool:
        """Try to solve coupled violations as square blocks.

        When multiple violated relations share the same output variable,
        they form a coupled system.  This method identifies such groups,
        constructs a square block of unknowns, and solves it using the
        least-squares block solver.

        Args:
            values: Current variable values.

        Returns:
            True if progress was made (a coupled block was solved).
        """
        violated = list(self._state["violated"])
        if not violated:
            return False

        # Group violated relations by preferred target variable.
        by_output: dict[str, list[object]] = {}
        for rel in violated:
            target_name = rel.preferred_target
            if target_name is None:
                continue
            by_output.setdefault(target_name, []).append(rel)

        for output, rels in by_output.items():
            if len(rels) < 2:
                continue

            # Identify candidate unknowns: computed (non-explicit-input)
            # variables that appear as inputs in these relations.
            unknowns = [output]
            candidates: dict[str, int] = {}
            for rel in rels:
                for var_name in rel.required_inputs(output):
                    if var_name == output:
                        continue
                    var = self._graph["vars"].get(var_name)
                    if var is not None and var.input_source == "explicit":
                        continue
                    if var is not None and var.fixed:
                        continue
                    candidates.setdefault(var_name, 0)
                    candidates[var_name] += 1

            # Sort by frequency (most connected first).
            sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
            needed = len(rels) - len(unknowns)
            for name, _ in sorted_candidates:
                if needed <= 0:
                    break
                unknowns.append(name)
                needed -= 1

            if len(unknowns) != len(rels):
                continue

            unknowns_sorted = sorted(unknowns)
            block_rels = rels[: len(unknowns_sorted)]
            solved = self._solve_block(block_rels, unknowns_sorted, values)
            if not solved:
                continue

            # Verify the block solution reduces violations.
            test_values = dict(values)
            test_values.update(solved)
            new_violations = self._violated_relations(test_values, rels)
            if len(new_violations) >= len(rels):
                continue  # No improvement

            # Set the values, bypassing per-variable tolerance so
            # that the coupled solution is accepted even when
            # individual changes are small.
            rel_names = [r.name for r in block_rels]
            for name, value in solved.items():
                self._set_value(
                    name,
                    value,
                    reason="coupled_solve",
                    relation=rel_names,
                    force=True,
                )

            log_message(self._log, logging.INFO, 
                "Resolved coupled violations for %s via block solve: %s",
                output,
                {k: f"{safe_float(v):.6g}" for k, v in solved.items()},
            )
            return True

        return False

    def solve(self) -> None:
        """Solve the relation system in-place by enforcing relations until stable."""
        rel_nodes = self.relations
        graph = self._graph
        rels_to_vars = graph["rels_to_vars"]
        var_order = graph["var_order"]
        rel_index = {rel: idx for idx, rel in enumerate(rel_nodes)}
        var_count = len(var_order)
        mode_overwrite = self.mode == "overwrite"
        log_message(self._log, logging.INFO, 
            "Starting solve: %s relations, %s variables, mode=%s, pass_mode=%s",
            len(rel_nodes),
            var_count,
            self.mode,
            mode_overwrite,
        )
        log_message(self._log, logging.DEBUG, 
            "solve() called: n_relations=%s, n_variables=%s, mode=%s",
            len(rel_nodes),
            var_count,
            self.mode,
        )
        # Step 0: check mode is read-only, so do not mutate variable values.
        if self.mode == "check":
            log_message(self._log, logging.DEBUG, "solve(): early exit for check mode")
            return  # Early exit for check mode

        self._state["volume_consistency_warned"].clear()

        # Step 1: seed the queue with all relations for the first pass.
        self._state["pending_rels"].update(rel_nodes)

        passes = 0
        log_message(self._log, logging.DEBUG, "solve(): entering main loop")
        while True:
            # Step 2: enforce pending relations and update violation set.
            log_message(self._log, logging.DEBUG, f"solve(): pass iteration {passes}")
            pass_label = self._state["pass_id"] if self._state["pass_id"] >= 0 else 0
            log_message(self._log, logging.INFO, 
                "Pass %s: enforcing %s pending relations",
                pass_label,
                len(self._state["pending_rels"]),
            )
            log_message(self._log, logging.DEBUG, 
                f"_execute_pass: pass_id={self._state['pass_id']}, {len(self._state['pending_rels'])} pending relations"
            )

            # 2.1) Enforce direct relation applications over pending queue.
            # Step 2.1.a: process all currently pending relations in solving order.
            pending = deque(sorted(self._state["pending_rels"], key=rel_index.get))
            self._state["pending_rels"].clear()
            applied = 0
            steps = 0
            iterations = 0
            touched: set[object] = set()
            while pending:
                rel = pending.popleft()
                touched.add(rel)
                steps += 1

                # Step 2.1.b: gather known values for this relation and detect unknowns.
                rel_values = {name: self._get_value(name) for name in rels_to_vars.get(rel, ())}
                missing_variables = [name for name in rels_to_vars.get(rel, ()) if rel_values.get(name) is None]

                # Step 2.1.c: apply relation forward/backward depending on missing variable count.
                target_name = rel.preferred_target
                relation_applied = False
                if missing_variables:
                    log_message(self._log, logging.DEBUG, 
                        "  Relation '%s': %s missing variables %s",
                        rel.name,
                        len(missing_variables),
                        missing_variables,
                    )
                    if len(missing_variables) == 1:
                        missing_var = missing_variables[0]

                        # 2.1.c.1) Direct forward evaluation when only output is missing.
                        if target_name is not None and missing_var == target_name:
                            try:
                                res = rel.evaluate(rel_values)
                                log_message(self._log, logging.DEBUG, "  Relation '%s': evaluated %s = %s", rel.name, target_name, res)
                            except Exception as e:
                                log_message(self._log, logging.DEBUG, "  Relation '%s': evaluation failed: %s", rel.name, e)
                                res = None

                            if res is not None:
                                output_is_profile = self._var_ndim(target_name) == 1
                                if output_is_profile:
                                    res_log = as_profile_array(res)
                                    if res_log is None:
                                        scalar = safe_float(res)
                                        if scalar is not None:
                                            res_log = np.asarray([scalar], dtype=float)
                                    if res_log is not None:
                                        log_message(self._log, logging.DEBUG, 
                                            "  Relation '%s': set profile output %s (n=%s)",
                                            rel.name,
                                            target_name,
                                            res_log.size,
                                        )
                                else:
                                    res_s = safe_float(res)
                                    if res_s is not None:
                                        log_message(self._log, logging.DEBUG, "  Relation '%s': set %s = %s", rel.name, target_name, res_s)

                                relation_applied = self._accept_candidate_values(
                                    {target_name: res},
                                    rels=[rel],
                                    reason="relation",
                                    relation=rel.name,
                                    protect_explicit=True,
                                    warn_input=True,
                                    check_violation_increase=False,
                                )
                                if not relation_applied:
                                    log_message(self._log, logging.DEBUG, "  Relation '%s': candidate rejected", rel.name)

                        # 2.1.c.2) Backward solve when one non-output variable is missing.
                        elif not (target_name is not None and missing_var != target_name and rel_values.get(target_name) is None):
                            log_message(self._log, logging.DEBUG, "    Attempting solve for %s", missing_var)
                            known_values = {
                                name: value
                                for name, value in rel_values.items()
                                if name != missing_var and value is not None
                            }
                            try:
                                solved_value = self._solve_for_value(rel, missing_var, known_values)
                            except Exception as e:
                                log_message(self._log, logging.DEBUG, "    ✗ Backward solve failed: %s", e)
                                log_message(self._log, logging.INFO, 
                                    "Backward solve failed for %s in '%s': %s",
                                    missing_var,
                                    rel.name,
                                    e,
                                )
                                solved_value = None

                            if solved_value is None:
                                log_message(self._log, logging.DEBUG, "    ✗ Backward solve returned None")
                            else:
                                solved_scalar = safe_float(solved_value)
                                if solved_scalar is None:
                                    log_message(self._log, logging.DEBUG, "    ✗ Backward solve returned non-finite value")
                                else:
                                    relation_applied = self._accept_candidate_values(
                                        {missing_var: solved_scalar},
                                        rels=[rel],
                                        reason="relation_inverse",
                                        relation=rel.name,
                                        protect_explicit=False,
                                        warn_input=False,
                                        check_violation_increase=True,
                                    )
                                    if relation_applied:
                                        log_message(self._log, logging.DEBUG, "    ✓ Backward solved %s = %.6g", missing_var, solved_scalar)
                                        log_message(self._log, logging.INFO, 
                                            "Solved: %s = %.4g from '%s'",
                                            missing_var,
                                            solved_scalar,
                                            rel.name,
                                        )
                                    else:
                                        log_message(self._log, logging.DEBUG, "    ✗ Backward solve rejected for %s", missing_var)
                else:
                    # 2.1.c.3) Full check/evaluate path when all variables are currently known.
                    target_var = self._graph["vars"].get(target_name) if target_name is not None else None
                    if (
                        target_name is not None
                        and target_var is not None
                        and target_var.input_source == "explicit"
                    ):
                        try:
                            target_expected = rel.evaluate(rel_values)
                        except Exception:
                            target_expected = None
                        target_expected_scalar = safe_float(target_expected)
                        target_actual_scalar = safe_float(rel_values.get(target_name))
                        target_rel_tol = (
                            rel.rel_tol_default
                            if rel.rel_tol_default is not None
                            else (target_var.rel_tol if target_var.rel_tol is not None else 0.0)
                        )
                        target_abs_tol = (
                            rel.abs_tol_default
                            if rel.abs_tol_default is not None
                            else (target_var.abs_tol if target_var.abs_tol is not None else 0.0)
                        )
                        target_violated = (
                            target_expected_scalar is not None
                            and target_actual_scalar is not None
                            and not within_tolerance(
                                target_actual_scalar,
                                target_expected_scalar,
                                rel_tol=target_rel_tol or 0.0,
                                abs_tol=target_abs_tol or 0.0,
                            )
                        )
                        if target_violated:
                            derived_candidates = [
                                name
                                for name in rels_to_vars.get(rel, ())
                                if name != target_name
                                and (var := self._graph["vars"].get(name)) is not None
                                and not var.fixed
                                and var.input_source != "explicit"
                            ]
                            if len(derived_candidates) == 1:
                                derived_name = derived_candidates[0]
                                known_values = {
                                    name: value
                                    for name, value in rel_values.items()
                                    if name != derived_name and value is not None
                                }
                                solved_value = self._solve_for_value(rel, derived_name, known_values)
                                solved_scalar = safe_float(solved_value)
                                if solved_scalar is not None:
                                    self._state["overrides"][derived_name] = {
                                        "value": solved_scalar,
                                        "relation": rel.name,
                                    }
                                    relation_applied = self._accept_candidate_values(
                                        {derived_name: solved_scalar},
                                        rels=[rel],
                                        reason="relation_inverse",
                                        relation=rel.name,
                                        protect_explicit=False,
                                        warn_input=False,
                                        check_violation_increase=False,
                                    )
                                    if relation_applied:
                                        log_message(
                                            self._log,
                                            logging.DEBUG,
                                            "  Relation '%s': synced %s = %s from explicit %s",
                                            rel.name,
                                            derived_name,
                                            solved_scalar,
                                            target_name,
                                        )
                                        continue
                    if target_name is not None:
                        out = target_name
                        try:
                            res = rel.evaluate(rel_values)
                            log_message(self._log, logging.DEBUG, "  Relation '%s': evaluated %s = %s", rel.name, out, res)
                        except Exception as e:
                            log_message(self._log, logging.DEBUG, "  Relation '%s': evaluation failed: %s", rel.name, e)
                            res = None

                        if res is not None:
                            output_is_profile = self._var_ndim(out) == 1
                            if output_is_profile:
                                res_log = as_profile_array(res)
                                if res_log is None:
                                    scalar = safe_float(res)
                                    if scalar is not None:
                                        res_log = np.asarray([scalar], dtype=float)
                                if res_log is not None:
                                    log_message(self._log, logging.DEBUG, 
                                        "  Relation '%s': set profile output %s (n=%s)",
                                        rel.name,
                                        out,
                                        res_log.size,
                                    )
                            else:
                                res_s = safe_float(res)
                                if res_s is not None:
                                    log_message(self._log, logging.DEBUG, "  Relation '%s': set %s = %s", rel.name, out, res_s)

                            relation_applied = self._accept_candidate_values(
                                {out: res},
                                rels=[rel],
                                reason="relation",
                                relation=rel.name,
                                protect_explicit=True,
                                warn_input=True,
                                check_violation_increase=False,
                            )
                            if not relation_applied:
                                log_message(self._log, logging.DEBUG, "  Relation '%s': candidate rejected", rel.name)

                if relation_applied:
                    applied += 1

                # Step 2.1.d: if new values created new pending relations, keep iterating.
                if not pending:
                    if not self._state["pending_rels"]:
                        break
                    pending = deque(sorted(self._state["pending_rels"], key=rel_index.get))
                    self._state["pending_rels"].clear()
                    iterations += 1
                    if iterations > 100:
                        log_message(self._log, logging.WARNING, "_execute_pass: breaking after %s iterations", iterations)
                        break
            if applied:
                log_message(self._log, logging.INFO, "Applied %s relation outputs", applied)
                log_message(self._log, logging.DEBUG, "_enforce_relations: applied %s relations", applied)
            log_message(self._log, logging.DEBUG, "_execute_pass: processed %s relations", steps)

            values = self._values_dict()
            self._warn_if_volume_integral_mismatch(values)
            violated_pending = self._violated_relations(values, list(touched))
            self._state["violated"].difference_update(touched)
            self._state["violated"].update(violated_pending)
            log_message(self._log, logging.INFO, "Pass %s: violated %s", pass_label, len(self._state["violated"]))
            log_message(self._log, logging.DEBUG, f"_execute_pass: {len(self._state['violated'])} violations")
            self._state["pending_rels"].clear()

            # 2.2) Group unknowns and solve square blocks (direct/inverse/least-squares).
            # Step 2.2.a: build unknown-variable signature map for each relation.
            unknown_map: dict[frozenset[str], list[object]] = {}
            for rel in rel_nodes:
                unknowns = frozenset(n for n in rels_to_vars.get(rel, ()) if n not in values)
                unknown_map.setdefault(unknowns, []).append(rel)
            log_message(self._log, logging.INFO, 
                "0-unknown relations: %s",
                len(unknown_map.get(frozenset(), [])),
            )
            # Step 2.2.b: solve square blocks from smallest to largest size.
            made_progress = False
            log_message(self._log, logging.DEBUG, "_solve_unknowns: trying block sizes 1..%s", self.n_max)
            for size in range(1, self.n_max + 1):
                candidates = [
                    (m, r)
                    for m, r in unknown_map.items()
                    if m and len(m) == size and len(r) >= size
                ]
                log_message(self._log, logging.INFO, "%sx%s blocks: %s", size, size, len(candidates))
                log_message(self._log, logging.DEBUG, "  %sx%s blocks: %s candidates", size, size, len(candidates))
                for unknown_set, rels in candidates:
                    unknowns = sorted(unknown_set)
                    rels_sorted = list(rels)
                    log_message(self._log, logging.DEBUG, "    Block with unknowns %s, %s relations", unknowns, len(rels))
                    for rel_subset in itertools.combinations(rels_sorted, size):
                        rel_names = [rel.name for rel in rel_subset]
                        log_message(self._log, logging.DEBUG, "      Trying relations: %s", rel_names)
                        solved = self._solve_block(list(rel_subset), unknowns, values)
                        if solved and self._accept_candidate_values(
                            solved,
                            rels=list(rel_subset),
                            reason="solve",
                            relation=rel_names,
                            values_map=values,
                            protect_explicit=False,
                            warn_input=False,
                            check_violation_increase=False,
                        ):
                            log_message(self._log, logging.DEBUG, "      ✓ Solved: %s", solved)
                            log_message(self._log, logging.INFO, 
                                "Solved block %s from relations %s",
                                unknowns,
                                ", ".join(rel_names),
                            )
                            made_progress = True
                            break
                        log_message(self._log, logging.DEBUG, "      ✗ Failed to solve or constraints violated")
                    if made_progress:
                        break
                if made_progress:
                    break
            if not made_progress:
                log_message(self._log, logging.DEBUG, "_solve_unknowns: no blocks solved")
            if made_progress or self._state["pending_rels"]:
                # New values or new pending relations mean another pass is required.
                continue

            # 2.3) Try coupled-violation block resolution (e.g. tau_E/P_loss).
            if self._state["violated"]:
                values = self._values_dict()
                if self._resolve_coupled_violations(values):
                    continue

            # Step 3: no progress; either override one culprit (overwrite mode) or finish.
            if self._state["violated"]:
                if mode_overwrite and passes < self.max_passes:
                    values = self._values_dict()
                    violated_all = self._violated_relations(values)
                    if violated_all:
                        self._state["violated"] = set(violated_all)
                    result = self._select_culprit(set(violated_all), values, rel_nodes)
                    if result:
                        kind, name, change, target, rel_name, meta = result
                        is_global = kind == "global" and meta is not None
                        if is_global:
                            sse_before = meta.get("sse_before") if meta else None
                            sse_after = meta.get("sse_after") if meta else None
                            score = meta.get("score") if meta else None
                            log_message(self._log, logging.INFO, 
                                "Global override candidate: %s -> %s (delta %.3g, sse %s -> %s, score %s)",
                                name,
                                target,
                                change,
                                f"{sse_before:.3g}" if sse_before is not None else "n/a",
                                f"{sse_after:.3g}" if sse_after is not None else "n/a",
                                f"{score:.3g}" if score is not None else "n/a",
                            )
                        else:
                            log_message(self._log, logging.INFO, 
                                "Override candidate: %s -> %s (delta %.3g) from %s",
                                name,
                                target,
                                change,
                                rel_name,
                            )
                        suffix = " (global)" if is_global else ""
                        log_message(self._log, logging.WARNING, 
                            "Inconsistency: overriding %s by %.3g -> %s%s",
                            name,
                            change,
                            target,
                            suffix,
                        )
                        self._state["overrides"][name] = {"value": target, "relation": rel_name}

                        # Restart a fresh pass seeded by the chosen override value.
                        if mode_overwrite:
                            self._start_pass()
                        else:
                            self._set_value(name, target, reason="override", relation=rel_name)
                        passes += 1
                        continue
                values = self._values_dict()
                self._warn_if_volume_integral_mismatch(values)
                self._warn_soft_constraints(values)
                return

            values = self._values_dict()
            self._warn_if_volume_integral_mismatch(values)
            self._warn_soft_constraints(values)
            return

    def evaluate(
        self,
        values: dict[str, object],
        *,
        chunk_size: int | None = None,
    ) -> dict[str, object]:
        """Evaluate relations on scalar or grid inputs using a dense matrix state.

        Args:
            values: Mapping of variable names to scalars, arrays, or profile payloads.
            chunk_size: Optional max row-chunk for relation evaluation calls.

        Returns:
            Updated values mapping containing computed outputs.
        """
        try:
            import numpy as np
        except Exception as exc:
            raise ImportError("evaluate requires numpy.") from exc

        eval_plan = self._state["eval_plan"]

        # Step 0: copy caller payload and check geometry-volume consistency once.
        evaluated = dict(values)
        self._state["volume_consistency_warned"].clear()
        self._warn_if_volume_integral_mismatch(evaluated)
        scalar_names = list(self._state["eval_scalar_order"])
        var_index = dict(self._state["eval_var_index"])

        # Step 1: register extra scalar keys that appear only in runtime payload.
        for name, value in evaluated.items():
            if name in var_index:
                continue
            if value is None or self._var_ndim(name) == 1:
                continue
            var_index[name] = len(scalar_names)
            scalar_names.append(name)

        # Step 2: infer the common broadcast shape used by dense matrix evaluation.
        candidate_arrays: list[np.ndarray] = []
        for name in scalar_names:
            raw = evaluated.get(name)
            if raw is None or self._var_ndim(name) == 1:
                continue
            try:
                arr = np.asarray(raw, dtype=float)
            except Exception:
                continue
            if arr.shape != ():
                candidate_arrays.append(arr)

        if candidate_arrays:
            target_shape = np.broadcast_arrays(*candidate_arrays)[0].shape
            n_points = int(np.prod(target_shape))
        else:
            target_shape = ()
            n_points = 1

        n_vars = len(scalar_names)
        state = np.full((n_points, n_vars), np.nan, dtype=float)
        known = np.zeros((n_points, n_vars), dtype=bool)

        # Step 3: seed matrix columns with known scalar values (broadcast as needed).
        for name, idx in var_index.items():
            raw = evaluated.get(name)
            if raw is None or self._var_ndim(name) == 1:
                continue
            try:
                arr = np.asarray(raw, dtype=float)
            except Exception:
                continue
            if arr.shape == ():
                state[:, idx] = float(arr)
                known[:, idx] = True
                continue
            try:
                flat = np.broadcast_to(arr, target_shape).reshape(-1)
            except Exception:
                continue
            state[:, idx] = flat
            known[:, idx] = True

        def _profile_to_scalar(value: object) -> float | None:
            """Convert scalar/profile payloads to one scalar for matrix evaluation."""
            arr = as_profile_array(value)
            if arr is not None:
                return float(np.mean(arr))
            return safe_float(value)

        # Step 4: iterate relation passes until no further writes are possible.
        max_iter = max(6, len(self.relations) + 1)
        row_step = max(1, chunk_size or n_points)
        for _ in range(max_iter):
            progress = False

            # 4.1) Resolve profile-target relations first.
            # In scan mode, profile outputs are reduced to one scalar per point
            # (profile mean) so scalar downstream relations can consume them.
            profile_progress = False
            for rel in self.relations:
                target_name = rel.preferred_target
                if target_name is None:
                    continue
                if self._var_ndim(target_name) != 1:
                    continue
                if evaluated.get(target_name) is not None:
                    continue

                input_values: dict[str, object] = {}
                valid = True
                for in_name in rel.required_inputs(target_name):
                    value = evaluated.get(in_name)
                    if value is None:
                        idx = var_index.get(in_name)
                        if idx is None:
                            valid = False
                            break
                        if n_points == 1:
                            if known[0, idx]:
                                value = float(state[0, idx])
                            else:
                                valid = False
                                break
                        else:
                            if np.all(known[:, idx]):
                                value = state[:, idx].copy()
                            else:
                                valid = False
                                break
                    input_values[in_name] = value
                if not valid:
                    continue

                if n_points == 1:
                    try:
                        result = rel.evaluate(input_values)
                    except Exception:
                        result = None
                    if result is None:
                        continue
                    evaluated[target_name] = result
                    profile_progress = True
                    continue

                out = np.full(n_points, np.nan, dtype=float)
                for row in range(n_points):
                    kwargs_row: dict[str, object] = {}
                    for in_name, value in input_values.items():
                        if isinstance(value, np.ndarray):
                            arr = np.asarray(value)
                            if arr.shape == (n_points,):
                                kwargs_row[in_name] = float(arr[row])
                            elif arr.shape == target_shape:
                                kwargs_row[in_name] = float(arr.reshape(-1)[row])
                            else:
                                kwargs_row[in_name] = arr
                        else:
                            kwargs_row[in_name] = value
                    try:
                        result = rel.evaluate(kwargs_row)
                    except Exception:
                        result = None
                    scalar = _profile_to_scalar(result)
                    if scalar is not None and np.isfinite(scalar):
                        out[row] = scalar
                if np.any(np.isfinite(out)):
                    evaluated[target_name] = out
                    profile_progress = True

            if profile_progress:
                progress = True
            # 4.2) Resolve scalar-target relations using vectorized path first,
            # then row-wise fallback when needed.
            for rel, output_idx0, input_names, input_idx0 in eval_plan:
                target_name = rel.preferred_target
                if target_name is None:
                    continue
                out_idx = var_index.get(target_name)
                if out_idx is None:
                    continue
                if output_idx0 is not None and output_idx0 != out_idx:
                    # Input-added scalar names can shift dynamic lookup only.
                    out_idx = var_index.get(target_name)
                pending = ~known[:, out_idx]
                if not np.any(pending):
                    continue

                scalar_inputs: list[tuple[str, int]] = []
                const_inputs: dict[str, object] = {}
                valid = True
                has_profile_const = False
                for in_name, in_idx in zip(input_names, input_idx0):
                    idx = in_idx if in_idx is not None else var_index.get(in_name)
                    if idx is not None:
                        pending &= known[:, idx]
                        scalar_inputs.append((in_name, idx))
                        continue

                    raw = evaluated.get(in_name)
                    if raw is None:
                        valid = False
                        break
                    if self._var_ndim(in_name) == 1:
                        if isinstance(raw, np.ndarray):
                            arr = np.asarray(raw)
                            if arr.shape == target_shape:
                                const_inputs[in_name] = arr.reshape(-1)
                            else:
                                const_inputs[in_name] = arr
                        else:
                            const_inputs[in_name] = raw
                        has_profile_const = True
                    elif isinstance(raw, np.ndarray):
                        try:
                            const_inputs[in_name] = np.broadcast_to(
                                np.asarray(raw, dtype=float), target_shape
                            ).reshape(-1)
                        except Exception:
                            valid = False
                            break
                    else:
                        const_inputs[in_name] = raw
                if not valid:
                    continue

                rows = np.flatnonzero(pending)
                if rows.size == 0:
                    continue

                wrote = False
                for start in range(0, rows.size, row_step):
                    row_ids = rows[start : start + row_step]
                    if not has_profile_const:
                        kwargs: dict[str, object] = {}
                        for in_name, idx in scalar_inputs:
                            kwargs[in_name] = state[row_ids, idx]
                        for in_name, value in const_inputs.items():
                            kwargs[in_name] = value[row_ids] if isinstance(value, np.ndarray) else value

                        try:
                            result = rel.evaluate(kwargs)
                            out = np.asarray(result, dtype=float)
                            if out.shape == ():
                                out = np.full(row_ids.size, float(out), dtype=float)
                            else:
                                out = np.broadcast_to(out, (row_ids.size,)).astype(float, copy=False)
                            state[row_ids, out_idx] = out
                            known[row_ids, out_idx] = True
                            wrote = True
                            continue
                        except Exception:
                            pass

                    out = np.full(row_ids.size, np.nan, dtype=float)
                    for i, row in enumerate(row_ids):
                        kwargs_row: dict[str, object] = {}
                        for in_name, idx in scalar_inputs:
                            kwargs_row[in_name] = float(state[row, idx])
                        for in_name, value in const_inputs.items():
                            if isinstance(value, np.ndarray):
                                if value.shape == (n_points,):
                                    kwargs_row[in_name] = float(value[row])
                                else:
                                    kwargs_row[in_name] = value
                            else:
                                kwargs_row[in_name] = value
                        try:
                            out[i] = float(rel.evaluate(kwargs_row))
                        except Exception:
                                out[i] = np.nan
                    finite = np.isfinite(out)
                    if np.any(finite):
                        state[row_ids[finite], out_idx] = out[finite]
                        known[row_ids[finite], out_idx] = True
                        wrote = True
                if wrote:
                    progress = True
            if not progress:
                break

        # Step 5: materialize computed matrix columns back into output mapping.
        for name, idx in var_index.items():
            if not known[:, idx].any() and name not in evaluated:
                continue
            if n_points == 1:
                evaluated[name] = None if not known[0, idx] else float(state[0, idx])
            else:
                evaluated[name] = state[:, idx].reshape(target_shape)

        return evaluated
    
    def _violated_relations(self, values: dict[str, object], rels: list | None = None) -> set[object]:
        """Return the set of violated relations for the given values."""
        rel_list = list(self.relations) if rels is None else rels
        return {rel for rel in rel_list if self._relation_status(rel, values)[0] == "VIOLATED"}

    def _relation_status(self, rel: object, values: dict[str, object]) -> tuple[str, float | None]:
        """Return (status, residual) for a relation given values."""
        values_scalar = self._to_scalar_values(values)
        if any(name not in values_scalar for name in self._graph["rels_to_vars"].get(rel, ())):
            return ("UNDECIDABLE", None)
        if self._constraints_violated(values, rel=rel):
            return ("VIOLATED", None)
        try:
            expected_scalar = safe_float(rel.evaluate(values))
        except Exception:
            expected_scalar = None
        if expected_scalar is None:
            return ("UNDECIDABLE", None)
        target_name = rel.preferred_target
        if target_name is None:
            return ("UNDECIDABLE", None)
        actual_scalar = safe_float(values_scalar.get(target_name))
        if actual_scalar is None:
            return ("UNDECIDABLE", None)
        residual = actual_scalar - expected_scalar
        rel_tol_val = rel.rel_tol_default or 0.0
        abs_tol_val = rel.abs_tol_default or 0.0
        status = "SAT" if within_tolerance(actual_scalar, expected_scalar, rel_tol=rel_tol_val, abs_tol=abs_tol_val) else "VIOLATED"
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
        rel_vars = self._graph["rels_to_vars"].get(rel, ())
        if any(values.get(name) is None for name in rel_vars):
            log_message(self._log, logging.DEBUG, "culprit_for_relation(%s): missing values for some variables", rel.name)
            return None

        explicit_vars = {
            name
            for name in rel_vars
            if (var := self._graph["vars"].get(name))
            and var.input_source == "explicit"
            and var.input_value is not None
        }

        log_message(self._log, logging.DEBUG, "culprit_for_relation(%s): analyzing %s explicit variables", rel.name, len(explicit_vars))
        best: tuple[str, float, float] | None = None
        try:
            exp_scalar = safe_float(rel.evaluate(values))
        except Exception:
            exp_scalar = None
        if exp_scalar is None:
            return None
        target_name = rel.preferred_target
        if target_name is None:
            return None
        act_scalar = safe_float(values.get(target_name))
        if act_scalar is None:
            return None
        residual = act_scalar - exp_scalar
        rel_tol = rel.rel_tol_default or 0.0
        abs_tol = rel.abs_tol_default or 0.0
        rel_violated = not within_tolerance(act_scalar, exp_scalar, rel_tol=rel_tol, abs_tol=abs_tol)

        # Prefer the explicit output as culprit when it's inconsistent.
        if target_name in explicit_vars and rel_violated:
            var = self._graph["vars"].get(target_name)
            if var is not None and not var.fixed:
                current_scalar = safe_float(var.input_value)
                if current_scalar is not None:
                    target_scalar = exp_scalar
                    merged = {**values, target_name: target_scalar}
                    if not self._constraints_violated(merged, rel=rel, names=[target_name]):
                        if current_scalar and target_scalar and (current_scalar * target_scalar) > 0:
                            change = abs(math.log10(abs(target_scalar / current_scalar)))
                        else:
                            scale = max(abs(current_scalar), abs(target_scalar), 1.0)
                            change = abs(target_scalar - current_scalar) / scale
                        return (target_name, change, target_scalar)

        for name in rel_vars:
            if name not in explicit_vars:
                continue

            var = self._graph["vars"].get(name)
            if var is None or var.fixed:
                continue

            current = safe_float(var.input_value)
            if current is None:
                continue

            target_scalar = None
            if name == target_name:
                target_scalar = exp_scalar
            elif name in rel.required_inputs():
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
                log_message(self._log, logging.DEBUG, "  New best culprit: %s, change=%.6g, target=%.6g", name, change, target_scalar)

        if best:
            log_message(self._log, logging.DEBUG, "culprit_for_relation(%s): best=%s, change=%.6g", rel.name, best[0], best[1])
        else:
            log_message(self._log, logging.DEBUG, "culprit_for_relation(%s): no culprit found", rel.name)
        return best

    
    def diagnose(self, values_override: dict[str, object] | None = None) -> dict[str, object]:
        """Return consolidated diagnostics for relations, variables, culprits, and soft constraints."""
        # Step 1: use provided values override or current effective values.
        values = values_override or self._values_dict()

        # Step 2: evaluate relation status and collect likely culprit suggestions.
        relation_results: list[tuple[str, str, float | None]] = []
        culprits: dict[str, tuple[str, float, float]] = {}
        violated_relations: list[str] = []
        for rel in self.relations:
            rel_name = getattr(
                rel,
                "name",
                rel.preferred_target,
            )
            status, residual = self._relation_status(rel, values)
            relation_results.append((rel_name, status, residual))
            if status == "VIOLATED":
                violated_relations.append(rel_name)
                culprit = self._culprit_for_relation(rel, values)
                if culprit is not None:
                    culprits[rel_name] = culprit

        # Step 3: evaluate variable consistency against input/current values.
        variable_issues: list[tuple[str, str, int | None]] = []
        for name in self._graph["var_order"]:
            var = self._graph["vars"].get(name)
            if var is None:
                continue
            if var.ndim == 1:
                variable_issues.append((name, "UNDETERMINABLE", None))
                continue
            input_val = var.input_value
            current_val = var.current_value
            if input_val is None or current_val is None:
                variable_issues.append((name, "UNDETERMINABLE", None))
                continue
            base = safe_float(input_val)
            cur = safe_float(current_val)
            if base is None or cur is None:
                variable_issues.append((name, "UNDETERMINABLE", None))
                continue
            rel_tol = var.rel_tol if var.rel_tol is not None else self.default_rel_tol
            abs_tol = var.abs_tol if var.abs_tol is not None else 0.0
            inconsistent = not within_tolerance(base, cur, rel_tol=rel_tol, abs_tol=abs_tol)
            variable_issues.append((name, "INCONSISTENT", 1) if inconsistent else (name, "CONSISTENT", None))

        # Step 4: evaluate soft-constraint violations without mutating state/warnings.
        values_scalar = self._to_scalar_values(values)

        soft_violations: list[tuple[str, str, str]] = []
        for name in self._graph["var_order"]:
            if name in self._state["enforced_var_names"]:
                continue
            cons = self._constraints["var_soft"].get(name, ())
            for constraint in self._constraint_violations(cons, values_scalar):
                soft_violations.append(("variable", name, constraint))
        for rel in self.relations:
            if rel in self._state["enforced_relations"]:
                continue
            rel_name = getattr(
                rel,
                "name",
                rel.preferred_target,
            )
            cons = self._constraints["rel_soft"].get(rel, ())
            for constraint in self._constraint_violations(cons, values_scalar):
                soft_violations.append(("relation", rel_name, constraint))

        # Step 5: return one consolidated diagnostic payload.
        return {
            "relation_status": relation_results,
            "violated_relations": violated_relations,
            "likely_culprits": culprits,
            "variable_issues": variable_issues,
            "soft_constraint_violations": soft_violations,
        }

    def export_relation_graph(self, path: str | Path = "relation_graph.html") -> Path:
        """Write an interactive HTML relation graph (variables as nodes, relations as edges)."""
        path = Path(path)
        values = self._values_dict()
        var_names: list[str] = []
        seen: set[str] = set()
        for name in self._graph["var_order"]:
            if name not in seen:
                seen.add(name)
                var_names.append(name)
        for rel in self.relations:
            for name in getattr(rel, "variables", {}):
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
            output = rel.preferred_target
            if output is None:
                continue
            for name in rel.required_inputs(output):
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
    
