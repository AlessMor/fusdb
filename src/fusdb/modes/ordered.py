"""Ordered mode."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fusdb.relation import Relation


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
                rels = [self._ordered_single_relation(entry) for entry in item]
                if not self._solve_ordered_block(rels, values, result):
                    result["executed_relations"] = executed
                    result["step_status"] = step_status
                    result["termination"] = "ordered evaluation stopped"
                    return result
                executed.extend(rel.name for rel in rels)
                for rel in rels:
                    record_status(rel, action="block")
                continue

            rel = self._ordered_single_relation(item)
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
