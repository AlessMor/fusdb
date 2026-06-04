"""Tests for reconcile culprit diagnostics."""

from __future__ import annotations

import pytest

from fusdb import Relation, RelationSystem, Variable
from fusdb.registry import RELATIONS


def test_reconcile_reports_adjusted_input_as_likely_culprit():
    """Ensure successful reconcile surfaces moved non-fixed inputs."""
    # Build a one-equation system whose free input must move to match a fixed output.
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

    # Reconcile should solve the system and rank the adjusted free input.
    result = system.reconcile()

    assert result["success"] is True
    assert result["likely_culprits"][0]["name"] == "H89"
    assert result["likely_culprits"][0]["status"] == "adjusted"
    assert result["likely_culprits"][0]["movement_score"] > 1.0
    assert all(item["name"] != "G89" for item in result["likely_culprits"])


def test_reconcile_excludes_fixed_variables_from_failed_culprit_ranking():
    """Ensure failed reconcile blames non-fixed supplied inputs instead of fixed constraints."""
    # Build two incompatible equations for the same fixed output.
    relations = [
        Relation(
            name="beta_from_h89",
            func=lambda H89: H89,
            input_names=("H89",),
            outputs=("beta",),
        ),
        Relation(
            name="beta_from_h89_plus_one",
            func=lambda H89: H89 + 1.0,
            input_names=("H89",),
            outputs=("beta",),
        ),
    ]
    system = RelationSystem(
        [Variable("H89", value=0.0), Variable("beta", value=0.0, fixed=True)],
        relations,
    )

    # Reconcile cannot satisfy both equations, so it should point back to H89 only.
    result = system.reconcile()

    assert result["success"] is False
    assert result["likely_culprits"][0]["name"] == "H89"
    assert result["likely_culprits"][0]["relation_count"] >= 1
    assert result["likely_culprits"][0]["max_relation_residual"] > 1.0
    assert result["likely_culprits"][0]["top_relations"]
    assert all(item["name"] != "beta" for item in result["likely_culprits"])
    assert any("Likely non-fixed input culprits" in warning for warning in result["warnings"])


def test_reconcile_normalizes_large_irrelevant_inputs():
    """Ensure large untouched inputs do not trigger premature xtol termination."""
    system = RelationSystem(
        [
            Variable("R", value=3.3, fixed=True),
            Variable("a", value=1.13, fixed=True),
            Variable("A", value=2.92, fixed=True),
            Variable("n_avg", value=1.3e20),
        ],
        [RELATIONS.get("Major radius"), RELATIONS.get("minor radius"), RELATIONS.get("Aspect ratio")],
    )

    result = system.reconcile()

    assert result["success"] is True
    assert system.variables_by_name["R_max"].value == pytest.approx(4.43)
    assert system.variables_by_name["R_min"].value == pytest.approx(2.17)


def test_reconcile_seeds_missing_output_scale_from_direct_relations():
    """Ensure missing outputs can be initialized from direct explicit relations."""
    system = RelationSystem(
        [
            Variable("I_p", value=7.8e6),
            Variable("n_avg", value=1.3e20),
            Variable("a", value=1.13, fixed=True),
            Variable("f_GW", value=0.67, fixed=True),
        ],
        [RELATIONS.get("Greenwald density limit"), RELATIONS.get("Greenwald density fraction")],
    )

    result = system.reconcile()

    assert result["success"] is True
    assert system.variables_by_name["n_GW"].value == pytest.approx(system.variables_by_name["n_avg"].value / 0.67, rel=1e-3)
