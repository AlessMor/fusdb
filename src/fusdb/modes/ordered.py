"""Ordered mode."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from fusdb.relation import Relation
from fusdb.relationsystem import Span
from fusdb.utils import scipy_bounds


def run(system: Any, order: Iterable[Any] | None = None, *, passes: int = 1, **_options: Any) -> dict[str, Any]:
    """Execute relations procedurally in the supplied order.

    Ordered mode is intentionally not a simultaneous verifier and not an
    optimizer.  At each step it uses the variable state available at that
    moment: immutable input values plus any current values written by previous
    steps.  If exactly one variable in the relation is missing, that variable is
    solved and its current value is overwritten.  If no variables are missing,
    the relation is only checked at that point.  Later steps may overwrite
    values and are not required to keep earlier relations satisfied.
    """
    self = system
    result = self._new_result("ordered")
    sequence = list(self.primary_relations if order is None else order)

    # Ordered mode operates on the current procedural state.  Current values
    # count as available values; completion/closure is deliberately disabled so
    # each ordered step is responsible for producing its own missing value.
    values = self._values_from_variables(
        for_solver=True,
        skip_missing=True,
        complete=False,
    )

    executed: list[str] = []
    step_status: list[dict[str, Any]] = []

    def record_status(rel: Relation, *, action: str) -> bool:
        status = self._verify_status(rel, self._relation_evaluation_values(rel, values))
        step_status.append({"relation": rel.name, "action": action, **status})
        return bool(status.get("verified", False))

    for _ in range(int(passes)):
        for item in sequence:
            if isinstance(item, (list, tuple)) and not isinstance(item, Relation):
                rels = [_ordered_single_relation(self, entry) for entry in item]
                if not _solve_ordered_block(self, rels, values, result):
                    result["executed_relations"] = executed
                    result["step_status"] = step_status
                    result["termination"] = "ordered evaluation stopped"
                    return result
                executed.extend(rel.name for rel in rels)
                for rel in rels:
                    record_status(rel, action="block")
                continue

            rel = _ordered_single_relation(self, item)
            executed.append(rel.name)
            known = {name: values[name] for name in rel.variables if name in values and values[name] is not None}
            missing = [name for name in rel.variables if name not in known]
            try:
                if not missing:
                    if not record_status(rel, action="verify"):
                        raise ValueError("relation not satisfied")
                    continue

                solved = rel.solve(known)
                if isinstance(solved, Mapping):
                    written = []
                    for name, value in solved.items():
                        values[name] = self._solver_value(name, value)
                        self.variables_by_name[name].set_value(self._public_value(name, value))
                        written.append(name)
                    action = "solve:" + ",".join(written)
                elif len(missing) == 1:
                    name = missing[0]
                    values[name] = self._solver_value(name, solved)
                    self.variables_by_name[name].set_value(self._public_value(name, solved))
                    action = f"solve:{name}"
                else:
                    raise ValueError(f"relation returned one value for multiple missing variables {missing}")

                if not record_status(rel, action=action):
                    raise ValueError("relation not satisfied after solve")
            except Exception as exc:
                result["errors"].append(f"Relation {rel.name!r} failed: {exc}")
                result["executed_relations"] = executed
                result["step_status"] = step_status
                result["termination"] = "ordered evaluation stopped"
                return result

    result.update(
        {
            "success": not result["errors"],
            "executed_relations": executed,
            "step_status": step_status,
            "termination": "ordered evaluation completed",
            "variables": self.variables_by_name,
            "values": self._values_from_variables(
                for_solver=False,
                skip_missing=True,
                complete=False,
            ),
        }
    )
    return result


def _ordered_single_relation(self: Any, item: Any) -> Relation:
    if isinstance(item, Relation):
        return item
    name = str(item)
    if name not in self.relations_by_name:
        raise KeyError(f"Unknown ordered relation {name!r}.")
    return self.relations_by_name[name]


def _solve_ordered_block(self: Any, rels: list[Relation], values: dict[str, Any], result: dict[str, Any]) -> bool:
    unknowns: list[str] = []
    for rel in rels:
        for name in rel.variables:
            self._ensure_variable_exists(name)
            if name not in values or values[name] is None:
                if self.variables_by_name[name].fixed:
                    result["errors"].append(f"Fixed variable {name!r} in ordered block has no value.")
                    return False
                if name not in unknowns:
                    unknowns.append(name)
    if not unknowns:
        return all(self._verify_status(rel, self._relation_evaluation_values(rel, values))["verified"] for rel in rels)

    # Primary path: delegate to the shared block solver used by reconcile's
    # initial computation (``_solve_initial_block``), which packs positive
    # unknowns logarithmically and refines starts with a coarse log-grid scan.
    # Using the same routine for ordered blocks and reconcile means an ordered
    # 2x2 block (e.g. solving tau_E/P_loss from the energy-confinement scaling
    # and the W_th = P_loss * tau_E balance) converges wherever reconcile does.
    # The local linear solve below is kept only as a fallback for structural
    # cases the shared solver declines (e.g. a profile-valued numerical core).
    block = self._solve_initial_block(tuple(unknowns), rels, values, residual_tol=1.0)
    if block is not None:
        for name, value in block["values"].items():
            values[name] = value
            self.variables_by_name[name].set_value(self._public_value(name, value))
        if all(self._verify_status(rel, self._relation_evaluation_values(rel, values))["verified"] for rel in rels):
            return True
        result["errors"].append("Ordered solve block did not verify.")
        return False

    spans: list[Span] = []
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for name in unknowns:
        var = self.variables_by_name[name]
        lb, ub = scipy_bounds(self.variable_registry.get(name).solver_domain, zero_tol=self.zero_tol)
        size = self._variable_dim(name)
        start = len(x0)
        offsets = []
        scales = []
        for i in range(size):
            try:
                init = float(self._initial_value(var, index=i if var.shape == 1 else None))
            except ValueError:
                known_start = self._block_start_from_knowns(rels, values, lb, ub)
                if known_start is None:
                    result["errors"].append(f"No initial value available for {name!r} in ordered block.")
                    return False
                init = known_start
            ref = self._reference_for_movement(var, init, index=i if var.shape == 1 else None)
            scale, offset, lo, hi, _transform = self._pack_scalar(name, var, init, lb, ub, scale_ref=ref, allow_log=False)
            x0.append(0.0)
            lower.append(lo)
            upper.append(hi)
            offsets.append(offset)
            scales.append(scale)
        spans.append(Span(name, start, len(x0), np.asarray(offsets), np.asarray(scales)))

    def block_values(x: np.ndarray) -> dict[str, Any]:
        out = dict(values)
        for name, start, stop, offsets, scales in spans:
            var = self.variables_by_name[name]
            actual = offsets + scales * x[start:stop]
            out[name] = actual.copy() if var.shape == 1 else float(actual[0])
        return out

    def residual(x: np.ndarray) -> np.ndarray:
        local = block_values(x)
        blocks = [self._residual_vector(rel, self._relation_evaluation_values(rel, local), safe=True) for rel in rels if rel.enforce]
        return np.concatenate(blocks) if blocks else np.empty(0)

    try:
        probe = residual(np.asarray(x0))
        if probe.size < len(x0):
            result["errors"].append(f"Ordered solve block is underdetermined: {probe.size} residuals for {len(x0)} unknowns {unknowns}.")
            return False
        sol = least_squares(residual, np.asarray(x0), bounds=(np.asarray(lower), np.asarray(upper)), method="trf", max_nfev=200)
    except Exception as exc:
        result["errors"].append(f"Ordered solve block failed: {exc}")
        return False
    solved = block_values(sol.x)
    if residual(sol.x).size and float(np.max(np.abs(residual(sol.x)))) > 1e-6:
        result["errors"].append("Ordered solve block did not verify.")
        return False
    for name in unknowns:
        values[name] = solved[name]
        self.variables_by_name[name].set_value(self._public_value(name, solved[name]))
    return True
