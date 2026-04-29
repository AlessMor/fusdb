from __future__ import annotations

from fusdb.popcon_class import Popcon
from fusdb.reactor_class import Reactor
from fusdb.relation_class import Relation
from fusdb.variable_class import Variable


def _scalar_var(name: str):
    """Return one scalar variable placeholder."""
    return Variable.make(name=name, ndim=0)


def test_reactor_popcon_returns_executed_popcon_object() -> None:
    """Expected: Reactor.popcon returns an already-run Popcon object with outputs and diagnostics."""
    rel_power = Relation.from_callable(
        name="test_popcon_power",
        target="P_fus",
        inputs=("T_avg", "n_avg"),
        func=lambda T_avg, n_avg: T_avg * n_avg,
    )
    rel_margin = Relation.from_callable(
        name="test_popcon_margin",
        target="greenwald_margin",
        inputs=("n_avg",),
        tags=("constraint",),
        func=lambda n_avg: n_avg - 1.0,
    )
    reactor = Reactor(
        id="R_popcon",
        relations=[rel_power, rel_margin],
        variables_dict={
            "T_avg": _scalar_var("T_avg"),
            "n_avg": _scalar_var("n_avg"),
            "P_fus": _scalar_var("P_fus"),
            "greenwald_margin": _scalar_var("greenwald_margin"),
        },
    )

    pop = reactor.popcon(
        scan_axes={"T_avg": [8.0, 10.0], "n_avg": [0.8, 1.2]},
        outputs=("P_fus", "greenwald_margin"),
    )

    assert isinstance(pop, Popcon)
    assert pop.axis_order == ["T_avg", "n_avg"]
    assert pop.allowed.shape == (2, 2)
    assert "P_fus" in pop.outputs_map
    assert "greenwald_margin" in pop.margins
    assert "fraction_allowed" in pop.diagnostics
    assert "violation_counts" in pop.diagnostics


def test_reactor_popcon_is_thin_wrapper_over_popcon(monkeypatch) -> None:
    """Expected: Reactor.popcon only forwards arguments to Popcon construction."""

    captured: dict[str, object] = {}

    class FakePopcon:
        """Capture construction args without running scan logic."""

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    reactor = Reactor(id="R_popcon_delegate", relations=[], variables_dict={})
    monkeypatch.setattr("fusdb.reactor_class.Popcon", FakePopcon)

    result = reactor.popcon(
        scan_axes={"x": [1.0, 2.0]},
        outputs=("P_fus",),
        constraints=("greenwald_margin",),
        exclude_constraints=("beta_margin",),
        where={"P_fus": (0.0, None)},
        chunk_size=64,
    )

    assert isinstance(result, FakePopcon)
    assert captured["reactor"] is reactor
    assert captured["scan_axes"] == {"x": [1.0, 2.0]}
    assert captured["outputs"] == ("P_fus",)
    assert captured["constraints"] == ("greenwald_margin",)
    assert captured["exclude_constraints"] == ("beta_margin",)
    assert captured["where"] == {"P_fus": (0.0, None)}
    assert captured["chunk_size"] == 64
