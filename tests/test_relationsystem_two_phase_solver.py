import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relation_class import Relation
from fusdb.relations.confinement.LHtransition import lh_transition_power
from fusdb.relations.operational_limits.density_limits import greenwald_density_fraction
from fusdb.relationsystem_class import RelationSystem
from fusdb.utils import relative_change
from fusdb.variable_class import Variable


def _effective_values(system: RelationSystem) -> dict[str, object]:
    """Return currently available values without relying on RelationSystem private state."""
    values: dict[str, object] = {}
    for name, var in system.variables_dict.items():
        value = var.current_value if var.current_value is not None else var.input_value
        if value is not None:
            values[name] = value
    return values


def _input_var(
    name: str,
    value: float,
    *,
    fixed: bool = False,
):
    """Build one scalar variable with explicit input metadata."""
    var = Variable.make(name=name, ndim=0)
    var.add_value(value, as_input=True)
    var.input_source = "explicit"
    var.fixed = fixed
    return var


def _empty_var(name: str):
    """Build one scalar variable without an input value."""
    return Variable.make(name=name, ndim=0)


def test_closure_block_solves_multi_output_connected_component():
    """Expected: closure block solving resolves unknowns in a multi-output connected component."""

    def _multi(y: float) -> dict[str, float]:
        return {"x": y + 1.0, "z": 2.0 * y}

    multi = Relation.from_callable(
        name="test_multi_bundle",
        outputs=("x", "z"),
        inputs=("y",),
        func=_multi,
    )
    total = Relation.from_callable(
        name="test_total_from_bundle",
        target="w",
        inputs=("x", "z"),
        func=lambda x, z: x + z,
    )

    variables = [
        _empty_var("y"),
        _empty_var("x"),
        _empty_var("z"),
        _input_var("w", 13.0),
    ]
    system = RelationSystem([multi, total], variables, mode="overwrite", max_passes=4)
    system.solve()

    assert system.variables_dict["y"].current_value == pytest.approx(4.0, rel=1e-5)
    assert system.variables_dict["x"].current_value == pytest.approx(5.0, rel=1e-5)
    assert system.variables_dict["z"].current_value == pytest.approx(8.0, rel=1e-5)
    assert system.last_result["stop_reason"] == "converged"


def test_canonical_graph_components_match_relation_variable_topology() -> None:
    """Expected: RelationSystem exposes one canonical graph for component discovery."""
    rel_a = Relation.from_callable(
        name="test_comp_x_from_a",
        target="x",
        inputs=("a",),
        func=lambda a: a + 1.0,
    )
    rel_b = Relation.from_callable(
        name="test_comp_y_from_b",
        target="y",
        inputs=("b",),
        func=lambda b: b + 1.0,
    )
    system = RelationSystem(
        [rel_a, rel_b],
        [_input_var("a", 1.0), _empty_var("x"), _input_var("b", 2.0), _empty_var("y")],
        mode="check",
    )

    components = system.graph.connected_components([rel_a, rel_b], ["a", "x", "b", "y"])
    component_names = sorted(
        (tuple(sorted(rel.name for rel in rels)), tuple(sorted(vars_)))
        for rels, vars_ in components
    )

    assert system.graph.relation_variable_names(rel_a) == ("a", "x")
    assert system.graph.relation_variable_names(rel_b) == ("b", "y")
    assert (("test_comp_x_from_a",), ("a", "x")) in component_names
    assert (("test_comp_y_from_b",), ("b", "y")) in component_names


def test_relationsystem_solving_order_separates_tag_groups_from_named_relations() -> None:
    """Expected: explicit relation names are emitted at their own ordered position."""
    minor = Relation.from_callable(
        name="minor radius",
        target="a",
        inputs=("R_max", "R_min"),
        func=lambda R_max, R_min: 0.5 * (R_max - R_min),
        tags=("geometry",),
    )
    major = Relation.from_callable(
        name="major radius",
        target="R",
        inputs=("R_max", "R_min"),
        func=lambda R_max, R_min: 0.5 * (R_max + R_min),
        tags=("geometry",),
    )
    plasma = Relation.from_callable(
        name="plasma balance",
        target="p",
        inputs=("R",),
        func=lambda R: R + 1.0,
        tags=("plasma",),
    )
    confinement = Relation.from_callable(
        name="confinement relation",
        target="tau",
        inputs=("p",),
        func=lambda p: p,
        tags=("confinement",),
    )

    system = RelationSystem(
        [minor, major, plasma, confinement],
        [],
        mode="check",
        solving_order=("geometry", "plasma", "minor radius"),
    )

    assert [rel.name for rel in system.relations] == [
        "major radius",
        "plasma balance",
        "minor radius",
    ]
    assert system.graph.relation_variable_ids


def test_profile_input_is_derived_from_scalar_avg_when_missing() -> None:
    """Expected: solver can derive missing profile input from explicit *_avg scalar."""

    relation = Relation.from_callable(
        name="test_profile_from_avg",
        target="x",
        inputs=("n_D",),
        func=lambda n_D: float(np.mean(n_D)),
    )
    n_d = Variable.make(name="n_D", ndim=1)
    system = RelationSystem(
        [relation],
        [
            _empty_var("x"),
            n_d,
            _input_var("n_D_avg", 4.2),
        ],
        mode="overwrite",
        max_passes=3,
    )
    system.solve()

    n_d = system.variables_dict["n_D"].current_value
    assert isinstance(n_d, np.ndarray)
    assert np.allclose(n_d, np.full(n_d.shape, 4.2))
    assert float(system.variables_dict["x"].current_value) == pytest.approx(4.2, rel=1e-9, abs=1e-9)


def test_scalar_avg_is_derived_from_profile_when_missing() -> None:
    """Expected: solver can derive missing *_avg scalar from explicit profile input."""

    relation = Relation.from_callable(
        name="test_avg_from_profile",
        target="y",
        inputs=("n_D_avg",),
        func=lambda n_D_avg: 2.0 * n_D_avg,
    )
    n_d = Variable.make(name="n_D", ndim=1)
    n_d.add_value(np.full(11, 3.0, dtype=float), as_input=True)
    n_d.input_source = "explicit"
    system = RelationSystem(
        [relation],
        [
            _empty_var("y"),
            _empty_var("n_D_avg"),
            n_d,
        ],
        mode="overwrite",
        max_passes=3,
    )
    system.solve()

    assert float(system.variables_dict["n_D_avg"].current_value) == pytest.approx(3.0, rel=1e-9, abs=1e-9)
    assert float(system.variables_dict["y"].current_value) == pytest.approx(6.0, rel=1e-9, abs=1e-9)


def test_closure_block_rejects_underdetermined_missing_symbol_blocks():
    """Expected: closure block LS does not assign guessed values when active equations are underdetermined."""

    # Both relations involve c, but n_max=2 forces a,b block candidates first.
    rel_a = Relation.from_callable(
        name="test_a_from_b_c",
        target="a",
        inputs=("b", "c"),
        func=lambda b, c: b + c,
    )
    rel_b = Relation.from_callable(
        name="test_b_from_a_c",
        target="b",
        inputs=("a", "c"),
        func=lambda a, c: a - c,
    )

    variables = [
        _empty_var("a"),
        _empty_var("b"),
        _empty_var("c"),
    ]
    system = RelationSystem([rel_a, rel_b], variables, mode="overwrite", n_max=2, max_passes=4)
    system.solve()

    # No variable should be fabricated from LS seeds (for example 1e-3).
    assert system.variables_dict["a"].current_value is None
    assert system.variables_dict["b"].current_value is None
    assert system.variables_dict["c"].current_value is None
    assert int(system.last_result["metrics"]["new_assignments_total"]) == 0


def test_closure_overwrites_inputs_when_relations_define_target():
    """Expected: closure prioritizes relation consistency over preserving explicit scalar inputs."""

    relation = Relation.from_callable(
        name="test_linear_balance",
        target="x",
        inputs=("y",),
        func=lambda y: 10.0 - y,
    )

    x = _input_var("x", 1.0)
    y = _input_var("y", 8.0)
    system = RelationSystem([relation], [x, y], mode="overwrite", max_passes=6)
    system.solve()

    x_now = float(system.variables_dict["x"].current_value)
    y_now = float(system.variables_dict["y"].current_value)
    assert x_now == pytest.approx(2.0, rel=1e-9, abs=1e-9)
    assert y_now == pytest.approx(8.0, rel=1e-9, abs=1e-9)
    assert system.last_result["stop_reason"] == "converged"


def test_stalled_stop_reason_when_no_adjustable_repair_exists():
    """Expected: solver reports a deterministic non-converged stop reason when constraints cannot be repaired."""

    constant_relation = Relation.from_callable(
        name="test_constant_x",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )

    fixed_x = _input_var("x", 1.0, fixed=True)
    system = RelationSystem([constant_relation], [fixed_x], mode="overwrite", max_passes=3)
    system.solve()

    assert system.last_result["stop_reason"] in {"stalled", "max_passes_reached"}
    diagnostics = system.diagnose()
    assert "test_constant_x" in diagnostics["violated_relations"]


def test_no_reconciliation_still_keeps_relation_consistent() -> None:
    """Expected: with max_passes=0, closure still updates values to satisfy directly solvable relations."""

    relation = Relation.from_callable(
        name="test_constant_target",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )

    x = _input_var("x", 1.0, fixed=False)
    system = RelationSystem([relation], [x], mode="overwrite", max_passes=0)
    system.solve()

    diagnostics = system.diagnose()
    assert "test_constant_target" not in diagnostics["violated_relations"]
    assert float(system.variables_dict["x"].current_value) == pytest.approx(2.0, rel=1e-9, abs=1e-9)


def test_multiwriter_conflict_uses_structural_reconciliation() -> None:
    """Expected: multi-writer scalar conflicts route through structural reconciliation instead of closure ping-pong."""
    rel_x_from_a = Relation.from_callable(
        name="test_x_from_a",
        target="x",
        inputs=("a",),
        func=lambda a: a,
    )
    rel_x_from_b = Relation.from_callable(
        name="test_x_from_b",
        target="x",
        inputs=("b",),
        func=lambda b: b,
    )

    x = _input_var("x", 0.0, fixed=False)
    a = _input_var("a", 1.0, fixed=True)
    b = _input_var("b", 2.0, fixed=True)
    system = RelationSystem(
        [rel_x_from_a, rel_x_from_b],
        [x, a, b],
        mode="overwrite",
        max_passes=4,
    )
    system.solve()

    diagnostics = system.diagnose()
    violated = set(diagnostics["violated_relations"])
    assert system.last_result["stop_reason"] != "cycle_detected"
    assert int(system.last_result["metrics"]["closure_multiwriter_relations"]) > 0
    assert int(system.last_result["metrics"]["reconciliation_direct_attempts"]) > 0
    assert violated.intersection({"test_x_from_a", "test_x_from_b"})


def test_culprit_prefers_smallest_relative_adjustment_over_output_rewrite():
    """Expected: culprit finder ranks by relative movement, not by output rewrite convenience."""

    relation = Relation.from_callable(
        name="test_product_balance",
        target="y",
        inputs=("a", "b"),
        func=lambda a, b: a * b,
    )

    a = _input_var("a", 1000.0, fixed=False)
    b = _input_var("b", 2.0, fixed=False)
    y = _input_var("y", 100.0, fixed=False)

    system = RelationSystem([relation], [a, b, y], mode="check", max_passes=0)
    diagnostics = system.diagnose()

    culprit_name, _change, _target = diagnostics["likely_culprits"]["test_product_balance"]
    assert culprit_name in {"a", "b"}


def test_inverse_fill_unknown_not_blocked_when_it_exposes_other_violations():
    """Expected: one-unknown inverse fills are accepted even when they reveal other violated relations."""
    n_gw = _input_var("n_GW", 6.5e19, fixed=False)
    f_gw = _input_var("f_GW", 1.2, fixed=False)
    n_avg = _empty_var("n_avg")
    b0 = _input_var("B0", 5.8, fixed=False)
    a_p = _input_var("A_p", 1200.0, fixed=False)
    p_lh = _input_var("P_LH", 1.0, fixed=False)

    system = RelationSystem(
        [greenwald_density_fraction, lh_transition_power],
        [n_gw, f_gw, n_avg, b0, a_p, p_lh],
        mode="overwrite",
        max_passes=0,
    )
    applied = system._run_relation_step(greenwald_density_fraction)

    expected_n_avg = 1.2 * 6.5e19
    assert applied is True
    n_avg_now = float(system.variables_dict["n_avg"].current_value)
    assert n_avg_now == pytest.approx(expected_n_avg, rel=1e-9)

    values = _effective_values(system)
    assert lh_transition_power in system._violated_relations(values, [lh_transition_power])


def test_small_scalar_inverse_does_not_accept_zero_bound_as_solution() -> None:
    """Expected: small nonzero targets invert to the root, not to the lower bound."""
    relation = Relation.from_callable(
        name="test_small_triangularity_inverse",
        target="delta",
        inputs=("delta_95",),
        func=lambda delta_95: 1.5 * delta_95,
    )
    delta = _input_var("delta", 0.003397448710900506, fixed=False)
    delta_95 = _empty_var("delta_95")
    system = RelationSystem([relation], [delta, delta_95], mode="overwrite")

    applied = system._run_relation_step(relation)

    assert applied is True
    assert system.variables_dict["delta_95"].current_value == pytest.approx(
        0.003397448710900506 / 1.5,
        rel=1e-9,
    )


def test_relative_change_penalizes_order_of_magnitude_drifts():
    """Expected: movement score grows with logarithmic scale drift, not only one-sided linear change."""
    huge_drift = relative_change(2.0e9, 4.0e-3)
    moderate_drift = relative_change(2.0e9, 1.2e9)

    assert huge_drift > moderate_drift
    assert huge_drift > 10.0


def test_culprit_selection_is_independent_of_input_source() -> None:
    """Expected: culprit ranking ignores input_source labels."""
    relation = Relation.from_callable(
        name="test_product_source_independence",
        target="y",
        inputs=("a", "b"),
        func=lambda a, b: a * b,
    )

    a_explicit = _input_var("a", 1000.0)
    b_default = _input_var("b", 2.0)
    y_value = _input_var("y", 100.0)
    a_explicit.input_source = "explicit"
    b_default.input_source = "default"
    system_a = RelationSystem([relation], [a_explicit, b_default, y_value], mode="check")
    culprit_a = system_a._culprit_for_relation(relation, _effective_values(system_a))

    a_default = _input_var("a", 1000.0)
    b_explicit = _input_var("b", 2.0)
    y_value_2 = _input_var("y", 100.0)
    a_default.input_source = "default"
    b_explicit.input_source = "explicit"
    system_b = RelationSystem([relation], [a_default, b_explicit, y_value_2], mode="check")
    culprit_b = system_b._culprit_for_relation(relation, _effective_values(system_b))

    assert culprit_a is not None and culprit_b is not None
    assert culprit_a[0] == culprit_b[0]
    assert culprit_a[1] == pytest.approx(culprit_b[1], rel=1e-12, abs=1e-12)


def test_reconciliation_adjusts_profile_variables_when_needed() -> None:
    """Expected: weighted reconciliation can move ndim=1 profile variables."""
    relation = Relation.from_callable(
        name="test_profile_balance",
        target="x",
        inputs=("p",),
        func=lambda p: float(np.mean(np.asarray(p, dtype=float))),
    )

    p = Variable.make(name="p", ndim=1)
    p.add_value(np.asarray([1.0, 1.0], dtype=float), as_input=True)
    p.input_source = "explicit"

    x = _input_var("x", 3.0, fixed=True)
    system = RelationSystem([relation], [p, x], mode="overwrite", max_passes=4, n_max=3)
    system.solve()

    p_now = np.asarray(system.variables_dict["p"].current_value, dtype=float)
    assert float(np.mean(p_now)) == pytest.approx(3.0, rel=1e-5, abs=5e-5)
    assert system._relation_status(relation, _effective_values(system))[0] == "SAT"


def test_reconciliation_direct_culprit_updates_profile_driven_output():
    """Expected: reconciliation can update explicit scalar outputs even when block LS skips profile relations."""
    relation = Relation.from_callable(
        name="test_profile_mean_output",
        target="x",
        inputs=("p",),
        func=lambda p: float(np.mean(np.asarray(p, dtype=float))),
    )

    p = Variable.make(name="p", ndim=1)
    p.add_value(np.asarray([2.0, 4.0], dtype=float), as_input=True)
    p.input_source = "explicit"
    x = _input_var("x", 1.0, fixed=False)

    system = RelationSystem([relation], [p, x], mode="overwrite", max_passes=6)
    system.solve()

    assert float(system.variables_dict["x"].current_value) == pytest.approx(3.0, rel=1e-8)
    assert system.last_result["stop_reason"] == "converged"


def test_final_check_summary_reports_satisfied_relations() -> None:
    """Expected: solve stores a terminal final-check summary when all relations are satisfied."""
    relation = Relation.from_callable(
        name="test_exact_constant",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )
    x = _input_var("x", 2.0, fixed=False)

    system = RelationSystem([relation], [x], mode="overwrite", max_passes=2)
    system.solve()

    final_check = system.last_result["final_check"]
    assert system.last_result["stop_reason"] == "converged"
    assert final_check["relations_checked"] == 1
    assert final_check["sat_count"] == 1
    assert final_check["violated_count"] == 0
    assert final_check["undecidable_count"] == 0
    assert final_check["all_satisfied"] is True
    assert int(system.last_result["metrics"]["final_check_sat_count"]) == 1


def test_final_check_summary_reports_undecidable_relations() -> None:
    """Expected: solve final-check marks unresolved relations as undecidable."""
    relation = Relation.from_callable(
        name="test_x_from_y",
        target="x",
        inputs=("y",),
        func=lambda y: y,
    )

    system = RelationSystem([relation], [_empty_var("x"), _empty_var("y")], mode="overwrite", max_passes=2)
    system.solve()

    final_check = system.last_result["final_check"]
    assert system.last_result["stop_reason"] == "final_check_undecidable"
    assert final_check["relations_checked"] == 1
    assert final_check["sat_count"] == 0
    assert final_check["violated_count"] == 0
    assert final_check["undecidable_count"] == 1
    assert final_check["all_satisfied"] is False
    assert int(system.last_result["metrics"]["final_check_undecidable_count"]) == 1


def test_final_check_summary_reports_violated_relations() -> None:
    """Expected: terminal verification flags violated relations on final values."""
    relation = Relation.from_callable(
        name="test_fixed_violation",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )
    x = _input_var("x", 1.0, fixed=True)

    system = RelationSystem([relation], [x], mode="overwrite", max_passes=2)
    system.solve()

    final_check = system.last_result["final_check"]
    assert final_check["relations_checked"] == 1
    assert final_check["sat_count"] == 0
    assert final_check["violated_count"] == 1
    assert final_check["undecidable_count"] == 0
    assert final_check["all_satisfied"] is False
    assert int(system.last_result["metrics"]["final_check_violated_count"]) == 1
    assert system.last_result["stop_reason"] != "converged"


def test_n0_relation_status_verifies_multi_output_relations() -> None:
    """Expected: n=0 relation verification marks inconsistent multi-output relations as violated."""

    def _bundle(y: float) -> dict[str, float]:
        return {"x": y + 1.0, "z": 2.0 * y}

    multi = Relation.from_callable(
        name="test_multi_output_check",
        outputs=("x", "z"),
        inputs=("y",),
        func=_bundle,
    )
    variables = [
        _input_var("y", 4.0),
        _input_var("x", 5.0),
        _input_var("z", 7.0),  # wrong: expected 8
    ]
    system = RelationSystem([multi], variables, mode="check")

    status, _residual = system._relation_status(multi, _effective_values(system))
    assert status == "VIOLATED"


def test_check_mode_solve_validates_without_writes_or_reconciliation() -> None:
    """Expected: check-mode solve validates current values without changing inputs or reconciling."""

    relation = Relation.from_callable(
        name="test_check_mode_constant",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )
    x = _input_var("x", 1.0, fixed=False)
    system = RelationSystem([relation], [x], mode="check", max_passes=6)

    system.solve()

    assert float(system.variables_dict["x"].current_value) == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert float(system.variables_dict["x"].input_value) == pytest.approx(1.0, rel=1e-12, abs=1e-12)

    final_check = system.last_result["final_check"]
    assert final_check["relations_checked"] == 1
    assert final_check["sat_count"] == 0
    assert final_check["violated_count"] == 1
    assert final_check["undecidable_count"] == 0
    assert final_check["all_satisfied"] is False
    assert system.last_result["stop_reason"] == "final_check_violated"

    metrics = system.last_result["metrics"]
    assert int(metrics["new_assignments_total"]) == 0
    assert int(metrics["reconciliation_success"]) == 0
    assert int(metrics["reconciliation_direct_attempts"]) == 0
    assert int(metrics["reconciliation_block_attempts"]) == 0


def test_variable_constraints_reject_invalid_solver_candidates() -> None:
    """Expected: Variable-owned constraints block invalid relation outputs."""
    relation = Relation.from_callable(
        name="test_negative_greenwald_fraction",
        target="f_GW",
        inputs=("x",),
        func=lambda x: -abs(x),
    )
    system = RelationSystem(
        [relation],
        [_input_var("x", 1.0), _empty_var("f_GW")],
        mode="overwrite",
    )
    system.solve()

    assert system.variables_dict["f_GW"].constraints == ("f_GW >= 0",)
    assert system.variables_dict["f_GW"].current_value is None


def test_self_updating_multi_output_step_requires_strict_improvement() -> None:
    """Expected: self-updating multi-output bundles are rejected when they do not improve the global objective."""

    def _self_bundle(x: float, y: float) -> dict[str, float]:
        return {"x": x + 1.0, "y": y}

    self_bundle = Relation.from_callable(
        name="test_self_bundle",
        outputs=("x", "y"),
        inputs=("x", "y"),
        func=_self_bundle,
    )
    anchor_x = Relation.from_callable(
        name="test_anchor_x",
        target="x",
        inputs=(),
        func=lambda: 1.0,
    )

    system = RelationSystem(
        [self_bundle, anchor_x],
        [_input_var("x", 1.0), _input_var("y", 0.0)],
        mode="overwrite",
        max_passes=2,
    )
    applied = system._run_relation_step(self_bundle)
    assert applied is False
    assert system.variables_dict["x"].current_value == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert system.variables_dict["y"].current_value == pytest.approx(0.0, rel=1e-12, abs=1e-12)


def test_diagnose_includes_structural_summary_partitions() -> None:
    """Expected: diagnostics expose compact structural partitions and block counts."""
    rel_a = Relation.from_callable(
        name="test_under_a",
        target="a",
        inputs=("b", "c"),
        func=lambda b, c: b + c,
    )
    rel_b = Relation.from_callable(
        name="test_under_b",
        target="b",
        inputs=("a", "c"),
        func=lambda a, c: a - c,
    )
    system = RelationSystem(
        [rel_a, rel_b],
        [_empty_var("a"), _empty_var("b"), _empty_var("c")],
        mode="check",
        n_max=2,
    )

    diagnostics = system.diagnose()
    assert "soft_constraint_violations" not in diagnostics
    summary = diagnostics["structural_summary"]
    assert "underconstrained_relations" in summary
    assert "wellconstrained_relations" in summary
    assert "overconstrained_relations" in summary
    assert "underconstrained_variables" in summary
    assert "wellconstrained_variables" in summary
    assert "overconstrained_variables" in summary
    assert "wellconstrained_block_count" in summary
    assert "violated_block_count" in summary
    assert "test_under_a" in set(summary["underconstrained_relations"] + summary["wellconstrained_relations"])
    assert "test_under_b" in set(summary["underconstrained_relations"] + summary["wellconstrained_relations"])


def test_solve_metrics_expose_structural_counters() -> None:
    """Expected: solve metrics include structural decomposition/reconciliation counters."""
    relation = Relation.from_callable(
        name="test_linear_counter",
        target="x",
        inputs=("y",),
        func=lambda y: 2.0 * y,
    )
    system = RelationSystem(
        [relation],
        [_empty_var("x"), _input_var("y", 3.0)],
        mode="overwrite",
        n_max=3,
        max_passes=2,
    )
    system.solve()

    metrics = system.last_result["metrics"]
    assert "structural_decompositions" in metrics
    assert "closure_structural_blocks" in metrics
    assert "closure_verification_passes" in metrics
    assert "reconciliation_direct_attempts" in metrics
    assert "reconciliation_block_attempts" in metrics
    assert "reconciliation_structural_components" in metrics
