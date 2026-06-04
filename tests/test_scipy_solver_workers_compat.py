"""Compatibility tests for SciPy least_squares solver options."""

from __future__ import annotations

from fusdb import Relation, RelationSystem, Variable
from fusdb.relationsystem import LEAST_SQUARES_SUPPORTS_WORKERS


def test_reconcile_ignores_workers_when_scipy_does_not_support_it():
    """Ensure reconcile mode remains usable across SciPy least_squares variants."""
    # Build a one-equation system with one free scalar to solve.
    relation = Relation(
        name="match_g89",
        func=lambda H89: H89,
        input_names=("H89",),
        outputs=("G89",),
    )
    system = RelationSystem(
        [Variable("H89", value=0.0), Variable("G89", value=1.0, fixed=True)],
        [relation],
    )

    # Request workers explicitly so the compatibility branch is exercised.
    result = system.reconcile(workers=map)

    # The solve should succeed regardless of the installed SciPy version.
    assert result["errors"] == []
    assert result["solver"]["success"] is True

    # Older SciPy builds should emit the fallback warning; newer ones should not.
    warning_fragment = "least_squares(workers=...)"
    if LEAST_SQUARES_SUPPORTS_WORKERS:
        assert all(warning_fragment not in warning for warning in result["warnings"])
    else:
        assert any(warning_fragment in warning for warning in result["warnings"])
