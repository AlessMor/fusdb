from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("bokeh")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bokeh.models import CustomJS, LayoutDOM

from fusdb.plotting.reactivity_plotter import (
    discover_reactivity_series,
    reactivity_plotter,
    save_reactivity_plotter_html,
)


def test_reactivity_plotter_discovers_the3_branch_relations():
    """Expected: branch-level THe3 relations are visible to the interactive plotter."""
    reactions = {series.reaction for series in discover_reactivity_series()}
    assert "THe3_D" in reactions
    assert "THe3_np" in reactions


def test_reactivity_plotter_returns_bokeh_layout():
    """Expected: the public plotter API returns a Bokeh layout object."""
    layout = reactivity_plotter(num_points=50)
    assert isinstance(layout, LayoutDOM)


def test_reactivity_plotter_checkbox_callback_updates_legend_visibility():
    """Expected: checkbox filters keep legend entries aligned with visible series."""
    layout = reactivity_plotter(num_points=30)
    plot = layout.children[0]
    reaction_selector = layout.children[1].children[1]

    assert len(plot.legend) == 1
    legend = plot.legend[0]
    assert len(legend.items) > 0

    callbacks = reaction_selector.js_property_callbacks["change:active"]
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert isinstance(callback, CustomJS)
    assert callback.args["legendItems"] == legend.items
    assert "legendItems[i].visible = isVisible;" in callback.code


def test_save_reactivity_plotter_html_writes_standalone_document(tmp_path: Path):
    """Expected: the docs helper writes an HTML file with Bokeh content."""
    output_path = tmp_path / "reactivity_plotter.html"
    saved = save_reactivity_plotter_html(output_path, num_points=30)
    assert saved == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "Fusion Reactivity Plotter" in html
    assert "Bokeh" in html
