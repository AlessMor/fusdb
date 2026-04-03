from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("bokeh")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bokeh.models import AutocompleteInput, Div, GraphRenderer, LayoutDOM, Plot

from fusdb.plotting.relation_graph import relation_graph_data, relation_graph_plotter, save_relation_graph_html


def _graph_renderer(layout: LayoutDOM) -> GraphRenderer:
    """Return the graph renderer inside one relation-graph layout.

    Args:
        layout: Layout returned by ``relation_graph_plotter``.

    Returns:
        Embedded graph renderer.
    """
    return next(ref for ref in layout.references() if isinstance(ref, GraphRenderer))


def test_relation_graph_data_is_not_empty() -> None:
    """Expected: relation graph data extraction discovers nodes and edges."""
    nodes, edges = relation_graph_data()
    assert len(nodes) > 0
    assert len(edges) > 0


def test_relation_graph_plotter_returns_layout_with_details_panel() -> None:
    """Expected: plotter returns graph plus search/details panels below it."""
    layout = relation_graph_plotter()
    assert isinstance(layout, LayoutDOM)
    assert hasattr(layout, "children")
    assert len(layout.children) == 2
    assert isinstance(layout.children[0], Plot)
    assert hasattr(layout.children[1], "children")
    assert len(layout.children[1].children) == 2

    renderer = _graph_renderer(layout)
    node_data = renderer.node_renderer.data_source.data
    edge_data = renderer.edge_renderer.data_source.data

    assert len(node_data["index"]) > 0
    assert len(edge_data["start"]) > 0
    assert "detail_html" in node_data
    assert "adjacent_html" in node_data
    assert "search_blob" in node_data
    assert "relation" in edge_data

    search_inputs = [ref for ref in layout.references() if isinstance(ref, AutocompleteInput)]
    assert len(search_inputs) == 1
    assert search_inputs[0].search_strategy == "includes"
    assert len(search_inputs[0].completions) > 0
    completion_set = {item.lower() for item in search_inputs[0].completions}
    assert "major_radius" in completion_set
    assert any("major radius" in item for item in completion_set)


def test_relation_graph_uses_variable_and_relation_node_markers() -> None:
    """Expected: variables render as circles and relations as boxes."""
    layout = relation_graph_plotter()
    renderer = _graph_renderer(layout)
    marker_set = set(renderer.node_renderer.data_source.data["marker"])

    assert "circle" in marker_set
    assert "square" in marker_set


def test_relation_graph_node_selection_callback_highlights_one_hop_neighbors() -> None:
    """Expected: node callback computes one-hop node/edge highlights and updates details."""
    layout = relation_graph_plotter()
    renderer = _graph_renderer(layout)
    node_source = renderer.node_renderer.data_source

    callbacks = node_source.selected.js_property_callbacks.get("change:indices", [])
    assert callbacks
    callback_code = callbacks[0].code

    assert "activeNodeIds" in callback_code
    assert "activeEdgeIndices" in callback_code
    assert "edgeSource.selected.indices = activeEdgeIndices" in callback_code
    assert "Adjacent Nodes" in callback_code


def test_relation_graph_layout_is_deterministic() -> None:
    """Expected: repeated renders keep stable node coordinates."""
    first_layout = relation_graph_plotter()
    second_layout = relation_graph_plotter()

    first_renderer = _graph_renderer(first_layout)
    second_renderer = _graph_renderer(second_layout)

    first_data = first_renderer.node_renderer.data_source.data
    second_data = second_renderer.node_renderer.data_source.data

    first_coords = {
        str(node_id): (float(x_coord), float(y_coord))
        for node_id, x_coord, y_coord in zip(
            first_data["index"],
            first_data["x"],
            first_data["y"],
            strict=True,
        )
    }
    second_coords = {
        str(node_id): (float(x_coord), float(y_coord))
        for node_id, x_coord, y_coord in zip(
            second_data["index"],
            second_data["x"],
            second_data["y"],
            strict=True,
        )
    }

    assert first_coords == second_coords


def test_save_relation_graph_html_writes_standalone_document(tmp_path: Path) -> None:
    """Expected: save helper writes standalone HTML output."""
    output_path = tmp_path / "relations_variables_graph.html"
    saved = save_relation_graph_html(output_path)
    assert saved == output_path

    html = output_path.read_text(encoding="utf-8")
    assert "fusdb Relation Graph" in html
    assert "Bokeh" in html
