"""Standalone Bokeh relation graph viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import html

import networkx as nx

from fusdb import relations
from fusdb.relation_util import _RELATION_REGISTRY
from fusdb.registry import load_allowed_variables


DEFAULT_DETAILS_HTML = (
    "<span style='color:#666'>Select a variable node or relation edge to see details.</span>"
)
NODE_COLOR = "#2b7a78"
NODE_MUTED_COLOR = "#d8e6e3"
EDGE_MUTED_COLOR = "#d0d7de"


@dataclass(frozen=True, slots=True)
class RelationGraphNode:
    name: str
    detail_html: str
    search_blob: str


@dataclass(frozen=True, slots=True)
class RelationGraphEdge:
    source: str
    target: str
    relation: str
    color: str
    detail_html: str
    search_blob: str


def _color_for(name: str) -> str:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def _format_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return ", ".join(f"{key}: {_format_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "[]"
        return ", ".join(_format_value(item) for item in value)
    if callable(value):
        return f"{value.__module__}.{value.__name__}"
    return str(value)


def _html_section(title: str, items: list[tuple[str, object]]) -> str:
    lines = [f"<b>{html.escape(title)}</b>"]
    for key, value in items:
        lines.append(f"<b>{html.escape(str(key))}</b>: {html.escape(_format_value(value))}")
    return "<br>".join(lines)


def _var_detail_html(name: str, spec: dict[str, object]) -> str:
    variable_items = [("name", name)]
    if "default_unit" in spec:
        variable_items.append(("unit", spec.get("default_unit")))
    if "ndim" in spec:
        variable_items.append(("ndim", spec.get("ndim")))

    detail = _html_section("Variable", variable_items)

    registry_items = [
        (key, spec.get(key))
        for key in ("aliases", "constraints", "description")
        if key in spec
    ]
    if registry_items:
        detail += "<br><br>" + _html_section("Registry", registry_items)
    return detail


def _relation_detail_html(relation) -> str:
    output = relation.preferred_target
    inputs = list(relation.required_inputs(output))
    relation_items = [
        ("name", relation.name),
        ("output", output),
        ("inputs", inputs),
        ("variables", list(relation.variables)),
        ("tags", list(relation.tags or ())),
        ("constraints", list(relation.constraints or ())),
        ("rel_tol_default", relation.rel_tol_default),
        ("abs_tol_default", relation.abs_tol_default),
        ("numeric_targets", list(relation.numeric_functions)),
        ("inverse_targets", list(relation.inverse_functions)),
        (
            "sympy_expression",
            str(relation.sympy_expression) if relation.sympy_expression is not None else None,
        ),
    ]
    return _html_section("Relation", relation_items)


def relation_graph_data() -> tuple[list[RelationGraphNode], list[RelationGraphEdge]]:
    """Return variable nodes and relation edges for the registered relation graph."""
    relations.import_relations()
    allowed_vars, _, _ = load_allowed_variables()
    relation_list = list(_RELATION_REGISTRY)

    variable_names = list(allowed_vars.keys())
    seen = set(variable_names)
    for relation in relation_list:
        for variable_name in relation.variables:
            if variable_name not in seen:
                seen.add(variable_name)
                variable_names.append(variable_name)

    nodes: list[RelationGraphNode] = []
    for name in variable_names:
        spec = allowed_vars.get(name, {}) or {}
        detail_html = _var_detail_html(name, spec)
        search_parts = [name]
        for value in spec.values():
            search_parts.append(str(value))
        nodes.append(
            RelationGraphNode(
                name=name,
                detail_html=detail_html,
                search_blob=" ".join(search_parts).lower(),
            )
        )

    edges: list[RelationGraphEdge] = []
    for relation in relation_list:
        output = relation.preferred_target
        if output is None:
            continue
        inputs = relation.required_inputs(output)
        detail_html = _relation_detail_html(relation)
        search_blob = " ".join(
            [
                relation.name,
                output,
                *inputs,
                *list(relation.tags or ()),
                *[str(item) for item in (relation.constraints or ())],
            ]
        ).lower()
        for source in inputs:
            edges.append(
                RelationGraphEdge(
                    source=source,
                    target=output,
                    relation=relation.name,
                    color=_color_for(relation.name),
                    detail_html=detail_html,
                    search_blob=search_blob,
                )
            )

    return nodes, edges


def relation_graph_plotter(*, width: int = 980, height: int = 860):
    """Return a standalone Bokeh layout for the relation/variable graph."""
    from bokeh.layouts import column, row
    from bokeh.models import Button, ColumnDataSource, CustomJS, Div, TextInput
    from bokeh.plotting import figure

    nodes, edges = relation_graph_data()

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.name)
    for edge in edges:
        graph.add_edge(edge.source, edge.target)

    if graph.number_of_nodes() == 0:
        positions: dict[str, tuple[float, float]] = {}
    else:
        positions = nx.spring_layout(graph, seed=13, k=2.4 / max(len(nodes) ** 0.5, 1.0))

    node_source = ColumnDataSource(
        data=dict(
            name=[node.name for node in nodes],
            x=[float(positions[node.name][0]) for node in nodes],
            y=[float(positions[node.name][1]) for node in nodes],
            detail_html=[node.detail_html for node in nodes],
            search_blob=[node.search_blob for node in nodes],
            color=[NODE_COLOR for _ in nodes],
            base_color=[NODE_COLOR for _ in nodes],
            muted_color=[NODE_MUTED_COLOR for _ in nodes],
            alpha=[0.95 for _ in nodes],
            line_alpha=[1.0 for _ in nodes],
            size=[12 for _ in nodes],
            base_size=[12 for _ in nodes],
        )
    )

    edge_source = ColumnDataSource(
        data=dict(
            xs=[[float(positions[edge.source][0]), float(positions[edge.target][0])] for edge in edges],
            ys=[[float(positions[edge.source][1]), float(positions[edge.target][1])] for edge in edges],
            source=[edge.source for edge in edges],
            target=[edge.target for edge in edges],
            relation=[edge.relation for edge in edges],
            detail_html=[edge.detail_html for edge in edges],
            search_blob=[edge.search_blob for edge in edges],
            color=[edge.color for edge in edges],
            base_color=[edge.color for edge in edges],
            muted_color=[EDGE_MUTED_COLOR for _ in edges],
            alpha=[0.60 for _ in edges],
            line_width=[1.8 for _ in edges],
            base_line_width=[1.8 for _ in edges],
        )
    )

    plot = figure(
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        active_scroll="wheel_zoom",
        title="Relations and Variables Graph",
        sizing_mode="stretch_width",
        toolbar_location="above",
    )
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.axis.visible = False
    plot.outline_line_color = "#d9d9d9"

    edge_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_alpha="alpha",
        line_width="line_width",
        source=edge_source,
    )
    node_renderer = plot.scatter(
        x="x",
        y="y",
        size="size",
        fill_color="color",
        line_color="color",
        fill_alpha="alpha",
        line_alpha="line_alpha",
        source=node_source,
    )

    details = Div(
        text=DEFAULT_DETAILS_HTML,
        width=360,
        height=height,
        styles={
            "overflow": "auto",
            "border-left": "1px solid #ddd",
            "padding": "0 0 0 14px",
            "font-size": "13px",
        },
    )
    search_input = TextInput(title="Search variables or relations", width=340)
    reset_button = Button(label="Reset view", button_type="default", width=120)
    hint = Div(
        text="Matches names first, then relation metadata and variable registry fields.",
        width=420,
        styles={"padding-top": "6px", "color": "#666"},
    )

    search_callback = CustomJS(
        args=dict(nodeSource=node_source, edgeSource=edge_source, searchInput=search_input),
        code="""
const query = searchInput.value.trim().toLowerCase();
const nodeData = nodeSource.data;
const edgeData = edgeSource.data;

for (let i = 0; i < nodeData.name.length; i++) {
  const hit = !query || nodeData.search_blob[i].includes(query) || nodeData.name[i].toLowerCase().includes(query);
  nodeData.color[i] = hit ? nodeData.base_color[i] : nodeData.muted_color[i];
  nodeData.alpha[i] = hit ? 0.95 : 0.18;
  nodeData.line_alpha[i] = hit ? 1.0 : 0.18;
  nodeData.size[i] = hit ? nodeData.base_size[i] : Math.max(8, nodeData.base_size[i] - 2);
}

for (let i = 0; i < edgeData.relation.length; i++) {
  const hit = !query || edgeData.search_blob[i].includes(query) || edgeData.relation[i].toLowerCase().includes(query);
  edgeData.color[i] = hit ? edgeData.base_color[i] : edgeData.muted_color[i];
  edgeData.alpha[i] = hit ? 0.75 : 0.10;
  edgeData.line_width[i] = hit ? edgeData.base_line_width[i] : 1.0;
}

nodeSource.change.emit();
edgeSource.change.emit();
""",
    )
    search_input.js_on_change("value", search_callback)

    node_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(details=details, nodeSource=node_source, edgeSource=edge_source, defaultHtml=DEFAULT_DETAILS_HTML),
            code="""
if (cb_obj.indices.length) {
  const index = cb_obj.indices[0];
  edgeSource.selected.indices = [];
  details.text = nodeSource.data.detail_html[index];
} else if (!edgeSource.selected.indices.length) {
  details.text = defaultHtml;
}
""",
        ),
    )
    edge_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(details=details, edgeSource=edge_source, nodeSource=node_source, defaultHtml=DEFAULT_DETAILS_HTML),
            code="""
if (cb_obj.indices.length) {
  const index = cb_obj.indices[0];
  nodeSource.selected.indices = [];
  details.text = edgeSource.data.detail_html[index];
} else if (!nodeSource.selected.indices.length) {
  details.text = defaultHtml;
}
""",
        ),
    )

    reset_button.js_on_click(
        CustomJS(
            args=dict(
                plot=plot,
                nodeSource=node_source,
                edgeSource=edge_source,
                details=details,
                searchInput=search_input,
                defaultHtml=DEFAULT_DETAILS_HTML,
            ),
            code="""
searchInput.value = "";
const nodeData = nodeSource.data;
const edgeData = edgeSource.data;
for (let i = 0; i < nodeData.name.length; i++) {
  nodeData.color[i] = nodeData.base_color[i];
  nodeData.alpha[i] = 0.95;
  nodeData.line_alpha[i] = 1.0;
  nodeData.size[i] = nodeData.base_size[i];
}
for (let i = 0; i < edgeData.relation.length; i++) {
  edgeData.color[i] = edgeData.base_color[i];
  edgeData.alpha[i] = 0.60;
  edgeData.line_width[i] = edgeData.base_line_width[i];
}
nodeSource.selected.indices = [];
edgeSource.selected.indices = [];
nodeSource.change.emit();
edgeSource.change.emit();
details.text = defaultHtml;
plot.reset.emit();
""",
        )
    )

    controls = row(search_input, reset_button, hint, sizing_mode="stretch_width")
    graph_column = column(controls, plot, sizing_mode="stretch_width")
    return row(graph_column, details, sizing_mode="stretch_width")


def save_relation_graph_html(path: str | Path, *, width: int = 980, height: int = 860) -> Path:
    """Write the standalone relation graph HTML file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_relation_graph_html(width=width, height=height),
        encoding="utf-8",
    )
    return output_path


def render_relation_graph_html(*, width: int = 980, height: int = 860) -> str:
    """Return the standalone relation graph HTML document."""
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(
        relation_graph_plotter(width=width, height=height),
        CDN,
        "fusdb Relation Graph",
    )


__all__ = (
    "RelationGraphEdge",
    "RelationGraphNode",
    "relation_graph_data",
    "relation_graph_plotter",
    "render_relation_graph_html",
    "save_relation_graph_html",
)
