"""Verify mode and final certification helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._common import result_from_certificate


def verify_values(system: Any, values: Mapping[str, Any], *, complete: bool = True) -> dict[str, Any]:
    """Verify one value map against every compiled enforced relation.

    This is the single certificate used by execution modes. It re-evaluates
    canonical Relation objects on the exact value map returned to the caller;
    optimizer termination is never used as a success condition.
    """
    self = system
    check_values = dict(values)
    if complete:
        check_values = self.complete(check_values)
    relation_status, residuals, errors, warnings = self.certify_relations(check_values)
    fixed_errors = self._fixed_value_errors(check_values)
    domain_errors = self._domain_errors(check_values)
    all_errors = [*errors, *fixed_errors, *domain_errors]
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
    values = self.solver_values()
    certificate = verify_values(self, values, complete=True)
    return result_from_certificate(self, "verify", certificate, termination="verification evaluated")
