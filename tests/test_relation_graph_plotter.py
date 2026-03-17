from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("bokeh")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bokeh.models import LayoutDOM

from fusdb.plotting.relation_graph import relation_graph_data, relation_graph_plotter, save_relation_graph_html


def test_relation_graph_data_is_not_empty():
    """Expected: the relation graph helper discovers variables and relation edges."""
    nodes, edges = relation_graph_data()
    assert len(nodes) > 0
    assert len(edges) > 0


def test_relation_graph_plotter_returns_bokeh_layout():
    """Expected: the public relation graph API returns a Bokeh layout object."""
    layout = relation_graph_plotter()
    assert isinstance(layout, LayoutDOM)


def test_save_relation_graph_html_writes_standalone_document(tmp_path: Path):
    """Expected: the docs helper writes an HTML file with Bokeh content."""
    output_path = tmp_path / "relations_variables_graph.html"
    saved = save_relation_graph_html(output_path)
    assert saved == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "fusdb Relation Graph" in html
    assert "Bokeh" in html
