"""Public reconcile/optimize/batch behavior and numerical regressions."""

from pathlib import Path

import pytest

import fusdb
from fusdb import Reactor, Variable


ROOT = Path(__file__).parents[1]
DEMO_YAML = ROOT / "reactors" / "DEMO_2022" / "reactor.yaml"
POLOMAC_YAML = ROOT / "reactors" / "Polomac" / "reactor.yaml"
ARC_YAML = ROOT / "reactors" / "ARC_V0" / "reactor.yaml"


def test_reconcile_rejects_invalid_movement_options():
    plan = Reactor.from_yaml(DEMO_YAML).relation_system().compile()
    result = plan.run("reconcile", movement_metric="decibels")

    assert result["termination"] == "invalid options"
    assert result["errors"]


def test_reconcile_reports_requested_movement_policy():
    plan = Reactor.from_yaml(DEMO_YAML).relation_system().compile()
    result = plan.run("reconcile", movement_objective="sum", movement_metric="absolute")

    assert result["solver"]["movement_objective"] == "sum"
    assert result["solver"]["movement_metric"] == "absolute"


def test_polomac_reconcile_preserves_known_physical_results():
    """Keep the previously diagnosed Polomac gap without allowing result drift."""
    result = Reactor.from_yaml(POLOMAC_YAML).run("reconcile")

    assert result["success"] is False
    assert result["failed_relations"] == ["Energy confinement balance"]
    values = result["values"]
    assert values["p_th"] == pytest.approx(3204.353268, rel=1e-9)
    assert values["W_th"] == pytest.approx(1.5 * 3204.353268 * 0.15, rel=1e-9)
    assert values["T_i_avg"] == pytest.approx(0.1, rel=1e-9)
    assert values["Z_eff"] == pytest.approx(1.0, rel=1e-12)
    assert values["P_aux"] == pytest.approx(7500.0, rel=1e-9)
    assert values["P_loss"] == pytest.approx(7500.0, rel=1e-9)


def test_optimize_accepts_constraints_and_initial_guesses():
    plan = Reactor.from_yaml(DEMO_YAML).relation_system().compile()
    result = plan.run(
        "optimize",
        objective="P_fus",
        sense="maximize",
        maxiter=1,
        constraints=[("P_fus", ">=", 2e9), ("P_in", ">=", "P_LH")],
        initial_guesses={"T_e_avg": 13.0},
    )

    assert result["mode"] == "optimize"
    assert result["termination"] != "invalid options"


def test_run_many_matches_direct_reconcile_result():
    cases = [{"delta": 0.3}]
    column = fusdb.run_many(str(ARC_YAML), cases, mode="reconcile", workers=1)[0]

    reactor = Reactor.from_yaml(ARC_YAML)
    reactor.add_variable(Variable("delta", value=0.3, fixed=True))
    direct = reactor.run("reconcile")

    assert column.result["success"] == direct["success"]
    assert column.result["failed_relations"] == direct["failed_relations"]
    for name in ("R", "a", "B_0", "I_p"):
        if name in direct.get("values", {}):
            assert column.result["values"][name] == pytest.approx(direct["values"][name])
