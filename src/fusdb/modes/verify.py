"""Verify mode and final certification helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def verify_values(system: Any, values: Mapping[str, Any], *, complete: bool = True) -> dict[str, Any]:
    """Verify one value map against every compiled enforced relation.

    This is the single certificate used by execution modes. It re-evaluates
    canonical Relation objects on the exact value map returned to the caller;
    optimizer termination is never used as a success condition.
    """
    self = system
    check_values = dict(values)
    if complete:
        check_values = self._complete_values(check_values, strict=False)
    relation_status, residuals, errors, warnings = self._evaluate_relation_residuals(
        check_values, strict=False, solver_residuals=False
    )
    fixed_errors = self._fixed_value_errors(check_values)
    domain_errors = self._domain_errors(check_values)
    compiler_errors = self._blocking_compiler_issues()
    all_errors = [*errors, *fixed_errors, *domain_errors, *compiler_errors]
    failed_relations = [
        name for name, status in relation_status.items()
        if status.get("enforced", True) and not status.get("verified", False)
    ]
    checked = {name for name, status in relation_status.items() if status.get("enforced", True)}
    expected = {rel.name for rel in self.relations if rel.enforce}
    missing = sorted(expected - checked)
    for name in missing:
        failed_relations.append(name)
        relation_status[name] = {
            "relation": name,
            "verified": False,
            "enforced": True,
            "errors": ["enforced relation was not checked"],
            "warnings": [],
        }
    max_residual = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    verified = not failed_relations and not all_errors
    return {
        "verified": bool(verified),
        "checked_relations": int(len(checked)),
        "expected_relations": int(len(expected)),
        "failed_relations": sorted(set(failed_relations)),
        "missing_checked_relations": missing,
        "max_residual": max_residual,
        "relation_status": relation_status,
        "residuals": residuals,
        "errors": all_errors,
        "warnings": warnings,
        "values": check_values,
    }



def run(system: Any, **_options: Any) -> dict[str, Any]:
    """Verify current public values against all compiled enforced relations."""
    self = system
    values = self._values_from_variables(for_solver=True, skip_missing=False)
    certificate = verify_values(self, values, complete=True)
    result = self._new_result("verify")
    result.update(
        {
            "relation_status": certificate["relation_status"],
            "residuals": certificate["residuals"].tolist(),
            "errors": certificate["errors"],
            "warnings": certificate["warnings"],
            "variable_status": self._classify_variables(certificate["relation_status"]),
            "termination": "verification evaluated",
            "success": bool(certificate["verified"]),
            "verified": bool(certificate["verified"]),
            "certificate": {k: v for k, v in certificate.items() if k not in {"residuals", "values"}},
            "variables": self.variables_by_name,
            "relations": self.primary_relations,
            "graph": self.graph,
            "compiler_report": self.compiler_report,
        }
    )
    return result
