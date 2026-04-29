from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.plotting import plot_cross_sections, plot_popcon
from fusdb.relation_class import Relation
from fusdb.reactor_class import Reactor
from fusdb.variable_class import Variable


def test_plot_cross_sections_module_and_reactor_method_return_axes():
    """Expected: cross-section plotting is available both as a module helper and as a Reactor method."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")
    reactor.solve()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    out_left = plot_cross_sections(reactor, ax=ax_left, label="ARC")
    out_right = reactor.plot_cross_sections(ax=ax_right, label="ARC")

    assert out_left is ax_left
    assert out_right is ax_right
    assert len(ax_left.lines) == 1
    assert len(ax_right.lines) == 1
    plt.close(fig)


def test_plot_popcon_module_and_reactor_method_return_axes():
    """Expected: POPCON plotting is available both as a module helper and as a Reactor method."""
    result = {
        "axes": {
            "T_avg": np.asarray([8.0, 10.0, 12.0], dtype=float),
            "n_avg": np.asarray([0.8e20, 1.0e20], dtype=float),
        },
        "axis_order": ["T_avg", "n_avg"],
        "outputs": {
            "P_fus": np.asarray([[100.0, 120.0], [150.0, 180.0], [210.0, 240.0]], dtype=float),
            "Q_sci": np.asarray([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=float),
        },
        "margins": {
            "greenwald_margin": np.asarray([[-0.2, -0.1], [-0.3, -0.2], [-0.4, -0.3]], dtype=float),
        },
        "allowed": np.asarray([[True, True], [True, True], [True, False]], dtype=bool),
    }

    reactor = Reactor()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))

    out_left = plot_popcon(
        result,
        x="T_avg",
        y="n_avg",
        fill="P_fus",
        contours=["Q_sci"],
        ax=ax_left,
    )
    out_right = reactor.plot_popcon(
        result,
        x="T_avg",
        y="n_avg",
        fill="P_fus",
        contours=["Q_sci"],
        ax=ax_right,
    )

    assert out_left is ax_left
    assert out_right is ax_right
    assert ax_left.get_title() == "POPCON: P_fus"
    assert ax_right.get_title() == "POPCON: P_fus"
    plt.close(fig)


def test_plot_popcon_accepts_popcon_object() -> None:
    """Expected: Reactor.plot_popcon accepts Popcon objects returned by Reactor.popcon."""
    rel_power = Relation.from_callable(
        name="test_plot_popcon_power",
        target="P_fus",
        inputs=("T_avg", "n_avg"),
        func=lambda T_avg, n_avg: T_avg * n_avg,
    )
    rel_margin = Relation.from_callable(
        name="test_plot_popcon_margin",
        target="greenwald_margin",
        inputs=("n_avg",),
        tags=("constraint",),
        func=lambda n_avg: n_avg - 1.0,
    )
    reactor = Reactor(
        relations=[rel_power, rel_margin],
        variables_dict={
            "T_avg": Variable.make(name="T_avg", ndim=0),
            "n_avg": Variable.make(name="n_avg", ndim=0),
            "P_fus": Variable.make(name="P_fus", ndim=0),
            "greenwald_margin": Variable.make(name="greenwald_margin", ndim=0),
        },
    )
    pop = reactor.popcon(
        scan_axes={"T_avg": [8.0, 10.0], "n_avg": [0.8, 1.2]},
        outputs=("P_fus", "greenwald_margin"),
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    out = reactor.plot_popcon(pop, x="T_avg", y="n_avg", fill="P_fus", ax=ax)

    assert out is ax
    assert ax.get_title() == "POPCON: P_fus"
    plt.close(fig)
