from __future__ import annotations

import fusdb

from fusdb import RelationSystem, Variable, aspect_ratio, plasma_loss_power


def test_top_level_relation_supports_simple_positional_forward_call() -> None:
    assert plasma_loss_power(120.0, 30.0) == 150.0
    assert plasma_loss_power(120.0, 30.0, P_loss=150.0)


def test_dynamic_relations_are_discoverable() -> None:
    assert fusdb.plasma_loss_power is plasma_loss_power
    assert "plasma_loss_power" in dir(fusdb)


def test_reconcile_persists_forward_completed_output() -> None:
    system = RelationSystem(
        [Variable("R", 3.0), Variable("a", 1.0), Variable("A")],
        [aspect_ratio],
    )

    result = system.reconcile()

    assert result["success"]
    assert system.values["A"] == 3.0


def test_optimize_no_dof_result_keeps_optimize_mode() -> None:
    system = RelationSystem(
        [
            Variable("R", 3.0, fixed=True),
            Variable("a", 1.0, fixed=True),
            Variable("A", 3.0, fixed=True),
        ],
        [aspect_ratio],
    )

    result = system.optimize(objective="A")

    assert result["mode"] == "optimize"
    assert result["success"]


def test_relationsystem_popcon_shortcut_delegates(monkeypatch) -> None:
    system = RelationSystem([], [])
    captured = {}

    def fake_run(mode, **options):
        captured.update(mode=mode, **options)
        return {"mode": mode}

    monkeypatch.setattr(system, "run", fake_run)

    result = system.popcon(x="R", y="a", outputs=("A",))

    assert result == {"mode": "popcon"}
    assert captured == {"mode": "popcon", "x": "R", "y": "a", "outputs": ("A",)}
