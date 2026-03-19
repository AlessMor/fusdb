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
    "<span style='color:#666'>Select a variable circle, relation box, or connection to see its description and metadata.</span>"
)
VARIABLE_FILL_COLOR = "#f3fbf9"
VARIABLE_LINE_COLOR = "#2b7a78"
VARIABLE_TEXT_COLOR = "#173f3b"
VARIABLE_MUTED_FILL_COLOR = "#eef2f3"
VARIABLE_MUTED_LINE_COLOR = "#c9d4d8"
VARIABLE_MUTED_TEXT_COLOR = "#8b9a9f"
RELATION_MUTED_COLOR = "#d0d7de"
EDGE_MUTED_COLOR = "#d0d7de"
VARIABLE_MARKER_SIZE = 44
VARIABLE_HIT_SIZE = 58
LAYOUT_COMPONENT_SPAN = 3
LAYOUT_COLUMN_GAP = 1.3
LAYOUT_ROW_GAP = 0.82
LAYOUT_ORDER_SWEEPS = 6
RELATION_BASE_WIDTH = 0.34
RELATION_WIDTH_PER_CHAR = 0.008
RELATION_MAX_WIDTH = 0.52
RELATION_BASE_HEIGHT = 0.16
RELATION_CONNECTOR_GAP = 0.11


@dataclass(frozen=True, slots=True)
class RelationGraphNode:
    name: str
    detail_html: str
    search_blob: str


@dataclass(frozen=True, slots=True)
class RelationGraphRelation:
    uid: str
    name: str
    output: str
    inputs: tuple[str, ...]
    color: str
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


def _relation_records() -> list[RelationGraphRelation]:
    relations.import_relations()
    relation_list = list(_RELATION_REGISTRY)

    records: list[RelationGraphRelation] = []
    for index, relation in enumerate(relation_list):
        output = relation.preferred_target
        if output is None:
            continue
        inputs = tuple(relation.required_inputs(output))
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
        records.append(
            RelationGraphRelation(
                uid=f"relation::{index}",
                name=relation.name,
                output=output,
                inputs=inputs,
                color=_color_for(relation.name),
                detail_html=detail_html,
                search_blob=search_blob,
            )
        )
    return records


def relation_graph_data() -> tuple[list[RelationGraphNode], list[RelationGraphEdge]]:
    """Return variable nodes and conceptual variable-to-variable relation edges."""
    allowed_vars, _, _ = load_allowed_variables()
    relation_records = _relation_records()

    variable_names = list(allowed_vars.keys())
    seen = set(variable_names)
    for record in relation_records:
        for variable_name in (*record.inputs, record.output):
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
    for record in relation_records:
        for source in record.inputs:
            edges.append(
                RelationGraphEdge(
                    source=source,
                    target=record.output,
                    relation=record.name,
                    color=record.color,
                    detail_html=record.detail_html,
                    search_blob=record.search_blob,
                )
            )

    return nodes, edges


def _connector_offsets(count: int, gap: float) -> list[float]:
    if count <= 1:
        return [0.0]
    span = gap * (count - 1)
    start = -0.5 * span
    return [start + gap * index for index in range(count)]


def _build_relation_digraph(
    variable_nodes: list[RelationGraphNode],
    relation_records: list[RelationGraphRelation],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in variable_nodes:
        graph.add_node(node.name, kind="variable", label=node.name)
    for record in relation_records:
        graph.add_node(record.uid, kind="relation", label=record.name)
        for input_name in record.inputs:
            graph.add_edge(input_name, record.uid)
        graph.add_edge(record.uid, record.output)
    return graph


def _component_layers(graph: nx.DiGraph) -> tuple[dict[str, int], dict[int, int]]:
    condensation = nx.condensation(graph)
    component_of: dict[str, int] = condensation.graph["mapping"]
    component_layer: dict[int, int] = {}
    for component in nx.topological_sort(condensation):
        predecessors = list(condensation.predecessors(component))
        if predecessors:
            component_layer[component] = max(component_layer[pred] + 1 for pred in predecessors)
        else:
            component_layer[component] = 0
    return component_of, component_layer


def _local_column(node_id: str, graph: nx.DiGraph, component_of: dict[str, int]) -> int:
    if graph.nodes[node_id]["kind"] == "relation":
        return 1

    same_component = component_of[node_id]
    has_same_component_relation_in = any(
        component_of[pred] == same_component and graph.nodes[pred]["kind"] == "relation"
        for pred in graph.predecessors(node_id)
    )
    has_same_component_relation_out = any(
        component_of[succ] == same_component and graph.nodes[succ]["kind"] == "relation"
        for succ in graph.successors(node_id)
    )

    if has_same_component_relation_out and not has_same_component_relation_in:
        return 0
    if has_same_component_relation_in and not has_same_component_relation_out:
        return 2
    if graph.in_degree(node_id) == 0:
        return 0
    if graph.out_degree(node_id) == 0:
        return 2
    return 0


def _deterministic_positions(
    variable_nodes: list[RelationGraphNode],
    relation_records: list[RelationGraphRelation],
) -> dict[str, tuple[float, float]]:
    graph = _build_relation_digraph(variable_nodes, relation_records)
    if graph.number_of_nodes() == 0:
        return {}

    component_of, component_layer = _component_layers(graph)
    columns: dict[int, list[str]] = {}
    column_of: dict[str, int] = {}
    for node_id in graph.nodes:
        column = (
            component_layer[component_of[node_id]] * LAYOUT_COMPONENT_SPAN
            + _local_column(node_id, graph, component_of)
        )
        column_of[node_id] = column
        columns.setdefault(column, []).append(node_id)

    def node_sort_key(node_id: str) -> tuple[int, str]:
        kind_priority = 0 if graph.nodes[node_id]["kind"] == "variable" else 1
        return (kind_priority, str(graph.nodes[node_id]["label"]).lower())

    ordered_columns = {
        column: sorted(node_ids, key=node_sort_key) for column, node_ids in columns.items()
    }
    sorted_columns = sorted(ordered_columns)
    undirected_graph = graph.to_undirected()

    def row_lookup() -> dict[str, int]:
        return {
            node_id: index
            for column in sorted_columns
            for index, node_id in enumerate(ordered_columns[column])
        }

    def reordered_nodes(column: int, neighbor_filter) -> list[str]:
        current_rows = row_lookup()
        scored: list[tuple[float, tuple[int, str], str]] = []
        for index, node_id in enumerate(ordered_columns[column]):
            neighbor_rows = [
                current_rows[neighbor]
                for neighbor in undirected_graph.neighbors(node_id)
                if neighbor_filter(column_of[neighbor], column)
            ]
            barycenter = sum(neighbor_rows) / len(neighbor_rows) if neighbor_rows else float(index)
            scored.append((barycenter, node_sort_key(node_id), node_id))
        scored.sort()
        return [node_id for _, _, node_id in scored]

    for _ in range(LAYOUT_ORDER_SWEEPS):
        for column in sorted_columns[1:]:
            ordered_columns[column] = reordered_nodes(column, lambda neighbor_col, own_col: neighbor_col < own_col)
        for column in reversed(sorted_columns[:-1]):
            ordered_columns[column] = reordered_nodes(column, lambda neighbor_col, own_col: neighbor_col > own_col)

    positions: dict[str, tuple[float, float]] = {}
    for column in sorted_columns:
        node_ids = ordered_columns[column]
        y_start = 0.5 * (len(node_ids) - 1) * LAYOUT_ROW_GAP
        for index, node_id in enumerate(node_ids):
            positions[node_id] = (
                float(column) * LAYOUT_COLUMN_GAP,
                y_start - index * LAYOUT_ROW_GAP,
            )

    return positions


def relation_graph_plotter(*, width: int = 980, height: int = 860):
    """Return a standalone Bokeh layout for the relation/variable graph."""
    from bokeh.layouts import column, row
    from bokeh.models import Button, ColumnDataSource, CustomJS, Div, TapTool, TextInput
    from bokeh.plotting import figure

    variable_nodes, _ = relation_graph_data()
    relation_records = _relation_records()

    positions = _deterministic_positions(variable_nodes, relation_records)
    relation_gap = RELATION_CONNECTOR_GAP

    variable_source = ColumnDataSource(
        data=dict(
            name=[node.name for node in variable_nodes],
            x=[float(positions[node.name][0]) for node in variable_nodes],
            y=[float(positions[node.name][1]) for node in variable_nodes],
            detail_html=[node.detail_html for node in variable_nodes],
            search_blob=[node.search_blob for node in variable_nodes],
            fill_color=[VARIABLE_FILL_COLOR for _ in variable_nodes],
            base_fill_color=[VARIABLE_FILL_COLOR for _ in variable_nodes],
            muted_fill_color=[VARIABLE_MUTED_FILL_COLOR for _ in variable_nodes],
            line_color=[VARIABLE_LINE_COLOR for _ in variable_nodes],
            base_line_color=[VARIABLE_LINE_COLOR for _ in variable_nodes],
            muted_line_color=[VARIABLE_MUTED_LINE_COLOR for _ in variable_nodes],
            text_color=[VARIABLE_TEXT_COLOR for _ in variable_nodes],
            base_text_color=[VARIABLE_TEXT_COLOR for _ in variable_nodes],
            muted_text_color=[VARIABLE_MUTED_TEXT_COLOR for _ in variable_nodes],
            fill_alpha=[0.98 for _ in variable_nodes],
            line_alpha=[1.0 for _ in variable_nodes],
            text_alpha=[1.0 for _ in variable_nodes],
            size=[VARIABLE_MARKER_SIZE for _ in variable_nodes],
            base_size=[VARIABLE_MARKER_SIZE for _ in variable_nodes],
            hit_size=[VARIABLE_HIT_SIZE for _ in variable_nodes],
        )
    )

    relation_widths: list[float] = []
    relation_heights: list[float] = []
    relation_hit_widths: list[float] = []
    relation_hit_heights: list[float] = []
    relation_xs: list[float] = []
    relation_ys: list[float] = []
    for record in relation_records:
        relation_xs.append(float(positions[record.uid][0]))
        relation_ys.append(float(positions[record.uid][1]))
        width_value = min(
            RELATION_MAX_WIDTH,
            RELATION_BASE_WIDTH + RELATION_WIDTH_PER_CHAR * len(record.name),
        )
        height_value = max(
            RELATION_BASE_HEIGHT,
            RELATION_BASE_HEIGHT + relation_gap * max(0, len(record.inputs) - 1),
        )
        relation_widths.append(width_value)
        relation_heights.append(height_value)
        relation_hit_widths.append(width_value * 1.18)
        relation_hit_heights.append(height_value * 1.18)

    relation_source = ColumnDataSource(
        data=dict(
            relation_id=[record.uid for record in relation_records],
            relation_name=[record.name for record in relation_records],
            output=[record.output for record in relation_records],
            x=relation_xs,
            y=relation_ys,
            width=relation_widths,
            height=relation_heights,
            hit_width=relation_hit_widths,
            hit_height=relation_hit_heights,
            detail_html=[record.detail_html for record in relation_records],
            search_blob=[record.search_blob for record in relation_records],
            fill_color=[record.color for record in relation_records],
            base_fill_color=[record.color for record in relation_records],
            muted_fill_color=[RELATION_MUTED_COLOR for _ in relation_records],
            line_color=[record.color for record in relation_records],
            base_line_color=[record.color for record in relation_records],
            muted_line_color=[RELATION_MUTED_COLOR for _ in relation_records],
            text_color=[record.color for record in relation_records],
            base_text_color=[record.color for record in relation_records],
            muted_text_color=[RELATION_MUTED_COLOR for _ in relation_records],
            fill_alpha=[0.16 for _ in relation_records],
            line_alpha=[1.0 for _ in relation_records],
            text_alpha=[0.95 for _ in relation_records],
            line_width=[2.2 for _ in relation_records],
            base_line_width=[2.2 for _ in relation_records],
        )
    )

    variable_positions = {
        node.name: (float(positions[node.name][0]), float(positions[node.name][1]))
        for node in variable_nodes
    }

    edge_xs: list[list[float]] = []
    edge_ys: list[list[float]] = []
    edge_relations: list[str] = []
    edge_relation_ids: list[str] = []
    edge_detail_html: list[str] = []
    edge_search_blob: list[str] = []
    edge_colors: list[str] = []
    for record, relation_x, relation_y, relation_width, relation_height in zip(
        relation_records,
        relation_xs,
        relation_ys,
        relation_widths,
        relation_heights,
        strict=True,
    ):
        input_offsets = _connector_offsets(len(record.inputs), relation_gap)
        for input_name, offset in zip(record.inputs, input_offsets, strict=True):
            input_x, input_y = variable_positions[input_name]
            edge_xs.append([input_x, relation_x - relation_width / 2.0])
            edge_ys.append([input_y, relation_y + offset])
            edge_relations.append(record.name)
            edge_relation_ids.append(record.uid)
            edge_detail_html.append(record.detail_html)
            edge_search_blob.append(record.search_blob)
            edge_colors.append(record.color)

        output_x, output_y = variable_positions[record.output]
        edge_xs.append([relation_x + relation_width / 2.0, output_x])
        edge_ys.append([relation_y, output_y])
        edge_relations.append(record.name)
        edge_relation_ids.append(record.uid)
        edge_detail_html.append(record.detail_html)
        edge_search_blob.append(record.search_blob)
        edge_colors.append(record.color)

    edge_source = ColumnDataSource(
        data=dict(
            xs=edge_xs,
            ys=edge_ys,
            relation=edge_relations,
            relation_id=edge_relation_ids,
            detail_html=edge_detail_html,
            search_blob=edge_search_blob,
            color=edge_colors,
            base_color=edge_colors,
            muted_color=[EDGE_MUTED_COLOR for _ in edge_colors],
            alpha=[0.72 for _ in edge_colors],
            line_width=[2.0 for _ in edge_colors],
            base_line_width=[2.0 for _ in edge_colors],
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
    plot.x_range.range_padding = 0.1
    plot.y_range.range_padding = 0.12

    edge_renderer = plot.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_alpha="alpha",
        line_width="line_width",
        source=edge_source,
    )

    relation_renderer = plot.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        fill_color="fill_color",
        line_color="line_color",
        fill_alpha="fill_alpha",
        line_alpha="line_alpha",
        line_width="line_width",
        source=relation_source,
    )
    relation_hit_renderer = plot.rect(
        x="x",
        y="y",
        width="hit_width",
        height="hit_height",
        fill_color="base_fill_color",
        line_color="base_line_color",
        fill_alpha=0.0,
        line_alpha=0.0,
        source=relation_source,
    )

    variable_renderer = plot.scatter(
        x="x",
        y="y",
        size="size",
        fill_color="fill_color",
        line_color="line_color",
        fill_alpha="fill_alpha",
        line_alpha="line_alpha",
        line_width=2.0,
        source=variable_source,
    )
    variable_hit_renderer = plot.scatter(
        x="x",
        y="y",
        size="hit_size",
        fill_color="base_fill_color",
        line_color="base_line_color",
        fill_alpha=0.0,
        line_alpha=0.0,
        source=variable_source,
    )

    plot.text(
        x="x",
        y="y",
        text="name",
        text_align="center",
        text_baseline="middle",
        text_color="text_color",
        text_alpha="text_alpha",
        text_font_size="8pt",
        source=variable_source,
    )
    plot.text(
        x="x",
        y="y",
        text="relation_name",
        text_align="center",
        text_baseline="middle",
        text_color="text_color",
        text_alpha="text_alpha",
        text_font_size="8pt",
        source=relation_source,
    )

    tap_tool = plot.select_one(TapTool)
    if tap_tool is not None:
        tap_tool.renderers = [variable_hit_renderer, relation_hit_renderer, edge_renderer]

    details = Div(
        text=DEFAULT_DETAILS_HTML,
        height=220,
        sizing_mode="stretch_width",
        styles={
            "overflow": "auto",
            "border-top": "1px solid #ddd",
            "padding": "12px 0 0 0",
            "font-size": "13px",
        },
    )
    search_input = TextInput(title="Search variables or relations", width=340)
    reset_button = Button(label="Reset view", button_type="default", width=120)
    hint = Div(
        text="Search matches variable names, relation names, and registry metadata.",
        width=420,
        styles={"padding-top": "6px", "color": "#666"},
    )

    search_callback = CustomJS(
        args=dict(
            variableSource=variable_source,
            relationSource=relation_source,
            edgeSource=edge_source,
            searchInput=search_input,
        ),
        code="""
const query = searchInput.value.trim().toLowerCase();
const variableData = variableSource.data;
const relationData = relationSource.data;
const edgeData = edgeSource.data;

for (let i = 0; i < variableData.name.length; i++) {
  const hit = !query || variableData.search_blob[i].includes(query) || variableData.name[i].toLowerCase().includes(query);
  variableData.fill_color[i] = hit ? variableData.base_fill_color[i] : variableData.muted_fill_color[i];
  variableData.line_color[i] = hit ? variableData.base_line_color[i] : variableData.muted_line_color[i];
  variableData.text_color[i] = hit ? variableData.base_text_color[i] : variableData.muted_text_color[i];
  variableData.fill_alpha[i] = hit ? 0.98 : 0.18;
  variableData.line_alpha[i] = hit ? 1.0 : 0.24;
  variableData.text_alpha[i] = hit ? 1.0 : 0.32;
  variableData.size[i] = hit ? variableData.base_size[i] : Math.max(34, variableData.base_size[i] - 10);
}

for (let i = 0; i < relationData.relation_name.length; i++) {
  const hit = !query || relationData.search_blob[i].includes(query) || relationData.relation_name[i].toLowerCase().includes(query);
  relationData.fill_color[i] = hit ? relationData.base_fill_color[i] : relationData.muted_fill_color[i];
  relationData.line_color[i] = hit ? relationData.base_line_color[i] : relationData.muted_line_color[i];
  relationData.text_color[i] = hit ? relationData.base_text_color[i] : relationData.muted_text_color[i];
  relationData.fill_alpha[i] = hit ? 0.16 : 0.05;
  relationData.line_alpha[i] = hit ? 1.0 : 0.20;
  relationData.text_alpha[i] = hit ? 0.95 : 0.28;
  relationData.line_width[i] = hit ? relationData.base_line_width[i] : 1.2;
}

for (let i = 0; i < edgeData.relation.length; i++) {
  const hit = !query || edgeData.search_blob[i].includes(query) || edgeData.relation[i].toLowerCase().includes(query);
  edgeData.color[i] = hit ? edgeData.base_color[i] : edgeData.muted_color[i];
  edgeData.alpha[i] = hit ? 0.72 : 0.08;
  edgeData.line_width[i] = hit ? edgeData.base_line_width[i] : 1.0;
}

variableSource.change.emit();
relationSource.change.emit();
edgeSource.change.emit();
""",
    )
    search_input.js_on_change("value", search_callback)

    variable_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                details=details,
                variableSource=variable_source,
                relationSource=relation_source,
                edgeSource=edge_source,
                defaultHtml=DEFAULT_DETAILS_HTML,
            ),
            code="""
if (cb_obj.indices.length) {
  const index = cb_obj.indices[0];
  relationSource.selected.indices = [];
  edgeSource.selected.indices = [];
  details.text = variableSource.data.detail_html[index];
} else if (!relationSource.selected.indices.length && !edgeSource.selected.indices.length) {
  details.text = defaultHtml;
}
""",
        ),
    )

    relation_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                details=details,
                variableSource=variable_source,
                relationSource=relation_source,
                edgeSource=edge_source,
                defaultHtml=DEFAULT_DETAILS_HTML,
            ),
            code="""
if (cb_obj.indices.length) {
  const index = cb_obj.indices[0];
  const relationName = relationSource.data.relation_name[index];
  const relationMatches = [];
  for (let i = 0; i < relationSource.data.relation_name.length; i++) {
    if (relationSource.data.relation_name[i] === relationName) {
      relationMatches.push(i);
    }
  }
  const edgeMatches = [];
  for (let i = 0; i < edgeSource.data.relation.length; i++) {
    if (edgeSource.data.relation[i] === relationName) {
      edgeMatches.push(i);
    }
  }
  const sameRelationSelection =
    cb_obj.indices.length === relationMatches.length &&
    cb_obj.indices.every((value, idx) => value === relationMatches[idx]);
  const sameEdgeSelection =
    edgeSource.selected.indices.length === edgeMatches.length &&
    edgeSource.selected.indices.every((value, idx) => value === edgeMatches[idx]);
  if (!sameRelationSelection || !sameEdgeSelection) {
    variableSource.selected.indices = [];
    relationSource.selected.indices = relationMatches;
    edgeSource.selected.indices = edgeMatches;
    return;
  }
  variableSource.selected.indices = [];
  details.text = relationSource.data.detail_html[index];
} else if (!variableSource.selected.indices.length && !edgeSource.selected.indices.length) {
  details.text = defaultHtml;
}
""",
        ),
    )

    edge_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                details=details,
                variableSource=variable_source,
                relationSource=relation_source,
                edgeSource=edge_source,
                defaultHtml=DEFAULT_DETAILS_HTML,
            ),
            code="""
if (cb_obj.indices.length) {
  const index = cb_obj.indices[0];
  const relationName = edgeSource.data.relation[index];
  const matches = [];
  for (let i = 0; i < edgeSource.data.relation.length; i++) {
    if (edgeSource.data.relation[i] === relationName) {
      matches.push(i);
    }
  }
  const relationMatches = [];
  for (let i = 0; i < relationSource.data.relation_name.length; i++) {
    if (relationSource.data.relation_name[i] === relationName) {
      relationMatches.push(i);
    }
  }
  const sameEdgeSelection =
    cb_obj.indices.length === matches.length &&
    cb_obj.indices.every((value, idx) => value === matches[idx]);
  const sameRelationSelection =
    relationSource.selected.indices.length === relationMatches.length &&
    relationSource.selected.indices.every((value, idx) => value === relationMatches[idx]);
  if (!sameEdgeSelection || !sameRelationSelection) {
    variableSource.selected.indices = [];
    relationSource.selected.indices = relationMatches;
    edgeSource.selected.indices = matches;
    return;
  }
  variableSource.selected.indices = [];
  details.text = edgeSource.data.detail_html[index];
} else if (!variableSource.selected.indices.length && !relationSource.selected.indices.length) {
  details.text = defaultHtml;
}
""",
        ),
    )

    reset_button.js_on_click(
        CustomJS(
            args=dict(
                plot=plot,
                variableSource=variable_source,
                relationSource=relation_source,
                edgeSource=edge_source,
                details=details,
                searchInput=search_input,
                defaultHtml=DEFAULT_DETAILS_HTML,
            ),
            code="""
searchInput.value = "";
const variableData = variableSource.data;
const relationData = relationSource.data;
const edgeData = edgeSource.data;

for (let i = 0; i < variableData.name.length; i++) {
  variableData.fill_color[i] = variableData.base_fill_color[i];
  variableData.line_color[i] = variableData.base_line_color[i];
  variableData.text_color[i] = variableData.base_text_color[i];
  variableData.fill_alpha[i] = 0.98;
  variableData.line_alpha[i] = 1.0;
  variableData.text_alpha[i] = 1.0;
  variableData.size[i] = variableData.base_size[i];
}

for (let i = 0; i < relationData.relation_name.length; i++) {
  relationData.fill_color[i] = relationData.base_fill_color[i];
  relationData.line_color[i] = relationData.base_line_color[i];
  relationData.text_color[i] = relationData.base_text_color[i];
  relationData.fill_alpha[i] = 0.16;
  relationData.line_alpha[i] = 1.0;
  relationData.text_alpha[i] = 0.95;
  relationData.line_width[i] = relationData.base_line_width[i];
}

for (let i = 0; i < edgeData.relation.length; i++) {
  edgeData.color[i] = edgeData.base_color[i];
  edgeData.alpha[i] = 0.72;
  edgeData.line_width[i] = edgeData.base_line_width[i];
}

variableSource.selected.indices = [];
relationSource.selected.indices = [];
edgeSource.selected.indices = [];
variableSource.change.emit();
relationSource.change.emit();
edgeSource.change.emit();
details.text = defaultHtml;
plot.reset.emit();
""",
        )
    )

    controls = row(search_input, reset_button, hint, sizing_mode="stretch_width")
    return column(plot, controls, details, sizing_mode="stretch_width")


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
    "RelationGraphRelation",
    "relation_graph_data",
    "relation_graph_plotter",
    "render_relation_graph_html",
    "save_relation_graph_html",
)
