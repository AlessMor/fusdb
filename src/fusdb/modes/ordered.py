"""Ordered mode."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fusdb.relation import Relation
from fusdb.seeding import solve_block
from fusdb.utils import ZERO_TOL

from ._common import new_result


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
    result = new_result(self, "ordered")
    sequence = list(self.primary_relations if order is None else order)

    # Ordered mode operates on the current procedural state.  Current values
    # count as available values; completion/closure is deliberately disabled so
    # each ordered step is responsible for producing its own missing value.
    values = self.solver_values()

    executed: list[str] = []
    step_status: list[dict[str, Any]] = []

    def record_status(rel: Relation, *, action: str) -> bool:
        status = self.relation_status_and_residual(rel, self.relation_evaluation_values(rel, values))[0]
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
                        values[name] = self.solver_value(name, value)
                        self.values[name] = self.public_value(name, value)
                        written.append(name)
                    action = "solve:" + ",".join(written)
                elif len(missing) == 1:
                    name = missing[0]
                    values[name] = self.solver_value(name, solved)
                    self.values[name] = self.public_value(name, solved)
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
            "values": self.public_values(),
        }
    )
    return result


def _ordered_single_relation(self: Any, item: Any) -> Relation:
    if isinstance(item, Relation):
        return item
    name = str(item)
    rel = self.relation_by_identifier(name)
    if rel is None:
        raise KeyError(f"Unknown ordered relation {name!r}.")
    return rel


def _solve_ordered_block(self: Any, rels: list[Relation], values: dict[str, Any], result: dict[str, Any]) -> bool:
    unknowns: list[str] = []
    for rel in rels:
        for name in rel.variables:
            self.track(name)
            if name not in values or values[name] is None:
                if name in self.fixed:
                    result["errors"].append(f"Fixed variable {name!r} in ordered block has no value.")
                    return False
                if name not in unknowns:
                    unknowns.append(name)
    if not unknowns:
        return all(self.relation_status_and_residual(rel, self.relation_evaluation_values(rel, values))[0]["verified"] for rel in rels)

    # Delegate to the shared block solver used by reconcile's initial
    # computation (:func:`fusdb.seeding.solve_block`), which packs positive
    # unknowns logarithmically and refines starts with a coarse log-grid scan.
    # Using the same routine for ordered blocks and reconcile means an ordered
    # 2x2 block (e.g. solving tau_E/P_loss from the energy-confinement scaling
    # and the W_th = P_loss * tau_E balance) converges wherever reconcile does.
    # Ordered blocks opt in to profile-valued numerical cores, which the
    # reconcile seeding path declines.
    block = solve_block(self, tuple(unknowns), rels, values, residual_tol=1.0, allow_profile_core=True)
    if block is None:
        result["errors"].append("Ordered solve block failed or did not verify.")
        return False
    for name, value in block.items():
        values[name] = value
        self.values[name] = self.public_value(name, value)
    if all(self.relation_status_and_residual(rel, self.relation_evaluation_values(rel, values))[0]["verified"] for rel in rels):
        return True
    result["errors"].append("Ordered solve block did not verify.")
    return False
