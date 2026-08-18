"""Broad public workflow contracts and numerical regressions.

These tests intentionally exercise complete user-facing workflows rather than
private solver helpers.  Detailed relation-wide safety checks live in
``test_relation_contract.py`` and reference-code regressions live in their
separate suites.
"""

from pathlib import Path

import pytest

import fusdb
from fusdb import Reactor, Variable


ROOT = Path(__file__).parents[1]
DEMO_YAML = ROOT / "reactors" / "DEMO_2022" / "reactor.yaml"
POLOMAC_YAML = ROOT / "reactors" / "Polomac" / "reactor.yaml"
ARC_YAML = ROOT / "reactors" / "ARC_V0" / "reactor.yaml"
SPARC_YAML = Path(__file__).parent / "cfspopcon_SPARC" / "reactor.yaml"


def test_reconcile_public_options_work_and_invalid_values_fail_loudly():
    plan = Reactor.from_yaml(DEMO_YAML).relation_system().compile()
    result = plan.run(
        "reconcile",
        exact=True,
        movement_objective="sum",
        movement_metric="absolute",
    )

    assert result["mode"] == "reconcile"
    assert result["termination"] != "invalid options"
    assert result["solver"]["movement_objective"] == "sum"
    assert result["solver"]["movement_metric"] == "absolute"

    bad = Reactor.from_yaml(DEMO_YAML).relation_system().compile().run(
        "reconcile", movement_metric="decibels"
    )
    assert bad["termination"] == "invalid options"
    assert bad["errors"]


def test_polomac_reconcile_preserves_known_physical_results():
    """A complete reduced-device solve must not drift to an unphysical branch."""
    result = Reactor.from_yaml(POLOMAC_YAML).run("reconcile")

    # Polomac still lacks a confinement-time determination.  Keep that known gap
    # explicit while protecting the physical values that are independently fixed.
    assert result["success"] is False
    assert result["failed_relations"] == ["Energy confinement balance"]
    values = result["values"]
    assert values["p_th"] == pytest.approx(3204.353268, rel=1e-9)
    assert values["W_th"] == pytest.approx(1.5 * 3204.353268 * 0.15, rel=1e-9)
    assert values["T_i_avg"] == pytest.approx(0.1, rel=1e-9)
    assert values["Z_eff"] == pytest.approx(1.0, rel=1e-12)
    assert values["P_aux"] == pytest.approx(7500.0, rel=1e-9)
    assert values["P_loss"] == pytest.approx(7500.0, rel=1e-9)


def test_optimize_supports_constraints_and_initial_guesses():
    plan = Reactor.from_yaml(DEMO_YAML).relation_system().compile()

    bad = plan.run(
        "optimize",
        objective="P_fus",
        sense="maximize",
        maxiter=1,
        constraints=[("P_fus", "!!", 1.0)],
    )
    assert bad["termination"] == "invalid options"

    result = Reactor.from_yaml(DEMO_YAML).relation_system().compile().run(
        "optimize",
        objective="P_fus",
        sense="maximize",
        maxiter=1,
        constraints=[("P_fus", ">=", 2e9), ("P_in", ">=", "P_LH")],
        initial_guesses={"T_e_avg": 13.0},
    )
    assert result["mode"] == "optimize"
    assert result["termination"] != "invalid options"


def test_run_many_preserves_every_mode_result():
    """Batch execution must return the same kind of result as direct execution."""
    arc = str(ARC_YAML)

    for mode in ("verify", "ordered", "reconcile"):
        column = fusdb.run_many(arc, [{"delta": 0.3}], mode=mode, workers=1)[0]
        reactor = Reactor.from_yaml(arc)
        reactor.add_variable(Variable("delta", value=0.3, fixed=True))
        direct = reactor.run(mode)
        assert set(column.result) == set(direct), mode
        assert column.result.get("success") == direct.get("success"), mode

    optimized = fusdb.run_many(
        arc,
        [{"delta": 0.3}],
        mode="optimize",
        workers=1,
        objective="P_fus",
        sense="maximize",
        maxiter=1,
    )[0]
    assert optimized.result["mode"] == "optimize"

    popcon = fusdb.run_many(
        str(SPARC_YAML),
        [{"B0": 12.2}],
        mode="popcon",
        workers=1,
        x={"variable": "average_electron_temp", "start": 8.0, "stop": 12.0, "num": 2},
        y={"variable": "average_electron_density", "start": 2.5e20, "stop": 3.0e20, "num": 2},
    )[0]
    assert popcon.result["success"] is True
    assert popcon.result["n_points"] == 4
    assert popcon.result["popcon"]["fields"]
