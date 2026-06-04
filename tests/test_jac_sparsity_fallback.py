"""Tests for Jacobian sparsity fallback behavior."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from fusdb import Relation, RelationSystem, Variable


def test_reconcile_ignores_wrong_shaped_jac_sparsity(monkeypatch):
    """Ensure SciPy solve falls back when the sparsity layout is inconsistent."""
    # Use two free variables so the solver attempts to build a sparsity pattern.
    relation = Relation(
        name="sum_to_beta",
        func=lambda H89, G89: H89 + G89,
        input_names=("H89", "G89"),
        outputs=("beta",),
    )
    system = RelationSystem(
        [
            Variable("H89", value=0.0),
            Variable("G89", value=0.0),
            Variable("beta", value=1.0, fixed=True),
        ],
        [relation],
    )

    # Force an invalid sparsity shape so the solve must discard it.
    monkeypatch.setattr(system, "_build_jac_sparsity", lambda *args, **kwargs: csr_matrix((1, 2), dtype=bool))

    # The solve should continue without jac_sparsity instead of failing early.
    result = system.reconcile()
    assert result["errors"] == []
    assert result["solver"]["success"] is True
    assert any("Ignoring jac_sparsity" in warning for warning in result["warnings"])
