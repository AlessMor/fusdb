from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("bokeh")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bokeh.models import ColumnDataSource, Div, GlyphRenderer, LayoutDOM

from fusdb.plotting.relation_graph import relation_graph_data, relation_graph_plotter, save_relation_graph_html


def test_relation_graph_data_is_not_empty():
    """Expected: the relation graph helper discovers variables and relation edges."""
    nodes, edges = relation_graph_data()
    assert len(nodes) > 0
    assert len(edges) > 0


def test_relation_graph_plotter_returns_bokeh_layout():
    """Expected: the public relation graph API returns a vertical layout with plot, controls, details, and uniform variable circles."""
    layout = relation_graph_plotter()
    assert isinstance(layout, LayoutDOM)
    assert hasattr(layout, "children")
    assert len(layout.children) == 3
    assert isinstance(layout.children[-1], Div)
    plot = layout.children[0]
    invisible_node_hits = [
        renderer
        for renderer in getattr(plot, "renderers", [])
        if isinstance(renderer, GlyphRenderer)
        and getattr(renderer.glyph, "fill_alpha", None) == 0.0
        and getattr(renderer.glyph, "line_alpha", None) == 0.0
    ]
    assert invisible_node_hits
    variable_sources = [
        source
        for source in layout.references()
        if isinstance(source, ColumnDataSource) and "name" in source.data and "size" in source.data
    ]
    assert len(variable_sources) == 1
    assert len(set(variable_sources[0].data["size"])) == 1


def test_relation_graph_edge_selection_expands_to_same_relation_name():
    """Expected: clicking one relation edge expands selection to every edge with the same relation name."""
    layout = relation_graph_plotter()
    edge_sources = [
        source
        for source in layout.references()
        if isinstance(source, ColumnDataSource) and "relation" in source.data
    ]
    assert len(edge_sources) == 1
    callbacks = edge_sources[0].selected.js_property_callbacks.get("change:indices", [])
    assert callbacks
    assert "relationName" in callbacks[0].code
    assert "edgeSource.selected.indices = matches" in callbacks[0].code


def test_relation_graph_layout_is_deterministic():
    """Expected: repeated renders keep the same node coordinates."""
    first_layout = relation_graph_plotter()
    second_layout = relation_graph_plotter()

    def sources(layout: LayoutDOM) -> tuple[ColumnDataSource, ColumnDataSource]:
        variable_source = next(
            source
            for source in layout.references()
            if isinstance(source, ColumnDataSource) and "name" in source.data and "size" in source.data
        )
        relation_source = next(
            source
            for source in layout.references()
            if isinstance(source, ColumnDataSource)
            and "relation_name" in source.data
            and "width" in source.data
        )
        return variable_source, relation_source

    first_variable_source, first_relation_source = sources(first_layout)
    second_variable_source, second_relation_source = sources(second_layout)

    assert first_variable_source.data["x"] == second_variable_source.data["x"]
    assert first_variable_source.data["y"] == second_variable_source.data["y"]
    assert first_relation_source.data["x"] == second_relation_source.data["x"]
    assert first_relation_source.data["y"] == second_relation_source.data["y"]


def test_save_relation_graph_html_writes_standalone_document(tmp_path: Path):
    """Expected: the docs helper writes an HTML file with Bokeh content."""
    output_path = tmp_path / "relations_variables_graph.html"
    saved = save_relation_graph_html(output_path)
    assert saved == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "fusdb Relation Graph" in html
    assert "Bokeh" in html
