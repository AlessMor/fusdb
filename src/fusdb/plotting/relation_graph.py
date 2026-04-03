"""Render the internal RelationSystem graph as an interactive Bokeh network."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import math
from pathlib import Path
import textwrap

import networkx as nx

from fusdb import relations
from fusdb.relationsystem_class import RelationSystem
from fusdb.relation_util import _RELATION_REGISTRY, relation_input_names
from fusdb.registry import load_allowed_variables
from fusdb.variable_util import make_variable


DEFAULT_DETAILS_HTML = (
    "<span style='color:#666'>Select one node to inspect its properties and immediate neighbors.</span>"
)
VARIABLE_FILL_COLOR = "#e9f6f3"
VARIABLE_LINE_COLOR = "#2b7a78"
VARIABLE_TEXT_COLOR = "#173f3b"
RELATION_FILL_COLOR = "#fff3df"
RELATION_LINE_COLOR = "#9a6500"
RELATION_TEXT_COLOR = "#2f2f2f"
EDGE_BASE_COLOR = "#c9ced6"
NODE_HIGHLIGHT_COLOR = "#ffb84d"
EDGE_HIGHLIGHT_COLOR = "#ff9f1c"
RELATION_LABEL_WIDTH = 16
RELATION_LABEL_MAX_LINES = 3


@dataclass(frozen=True, slots=True)
class RelationGraphNode:
    """Variable node summary returned by ``relation_graph_data``.

    Attributes:
        name: Variable name.
        detail_html: HTML details for docs/tests.
        search_blob: Lower-cased searchable text.
    """

    name: str
    detail_html: str
    search_blob: str


@dataclass(frozen=True, slots=True)
class RelationGraphEdge:
    """Variable-to-variable edge summary returned by ``relation_graph_data``.

    Attributes:
        source: Input variable name.
        target: Output variable name.
        relation: Relation name that defines the mapping.
        color: Deterministic relation color.
        detail_html: HTML relation details for docs/tests.
        search_blob: Lower-cased searchable text.
    """

    source: str
    target: str
    relation: str
    color: str
    detail_html: str
    search_blob: str


def _color_for(name: str) -> str:
    """Return one deterministic color from a relation name.

    Args:
        name: Relation name.

    Returns:
        Hex color string.
    """
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def _format_value(value: object) -> str:
    """Convert metadata values to readable text.

    Args:
        value: Metadata value.

    Returns:
        Plain-text representation.
    """
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
    """Build one compact HTML section.

    Args:
        title: Section title.
        items: Key/value rows.

    Returns:
        HTML fragment.
    """
    lines = [f"<b>{html.escape(title)}</b>"]
    for key, value in items:
        lines.append(f"<b>{html.escape(str(key))}</b>: {html.escape(_format_value(value))}")
    return "<br>".join(lines)


def _relation_label(name: str) -> str:
    """Return a wrapped relation label suitable for node text.

    Args:
        name: Relation name.

    Returns:
        Label with newline breaks.
    """
    normalized = " ".join(name.replace("_", " ").split())
    lines = textwrap.wrap(
        normalized,
        width=RELATION_LABEL_WIDTH,
        break_long_words=True,
        break_on_hyphens=True,
    )
    if not lines:
        return normalized
    if len(lines) <= RELATION_LABEL_MAX_LINES:
        return "\n".join(lines)

    leading = lines[: RELATION_LABEL_MAX_LINES - 1]
    trailing = " ".join(lines[RELATION_LABEL_MAX_LINES - 1 :])
    trailing = textwrap.shorten(trailing, width=RELATION_LABEL_WIDTH, placeholder="...")
    return "\n".join([*leading, trailing])


def _relation_signature(relation) -> str:
    """Return a compact signature string for one relation.

    Args:
        relation: Relation object.

    Returns:
        Signature text.
    """
    inputs = ", ".join(relation.inputs)
    outputs = ", ".join(relation.outputs)
    return f"{relation.name}({inputs}) -> {outputs}"


def _variable_detail_html(name: str, spec: dict[str, object], variable) -> str:
    """Build HTML details for one variable node.

    Args:
        name: Variable name.
        spec: Registry metadata for the variable.
        variable: ``Variable`` object from the relation system.

    Returns:
        HTML details block.
    """
    # Include identity and runtime defaults first.
    variable_items = [("name", name)]
    if variable is not None:
        variable_items.extend(
            [
                ("ndim", variable.ndim),
                ("unit", variable.unit),
                ("method", variable.method),
                ("rel_tol", variable.rel_tol),
                ("abs_tol", variable.abs_tol),
                ("fixed", variable.fixed),
            ]
        )
    else:
        variable_items.extend(
            [
                ("ndim", spec.get("ndim")),
                ("unit", spec.get("default_unit")),
                ("method", spec.get("method")),
            ]
        )

    detail = _html_section("Variable", variable_items)

    # Add registry text fields when present.
    registry_items = [
        (key, spec.get(key))
        for key in ("aliases", "constraints", "soft_constraints", "description")
        if key in spec
    ]
    if registry_items:
        detail += "<br><br>" + _html_section("Registry", registry_items)
    return detail


def _relation_detail_html(relation) -> str:
    """Build HTML details for one relation node.

    Args:
        relation: Relation object.

    Returns:
        HTML details block.
    """
    # Summarize signature and solver defaults.
    relation_items = [
        ("name", relation.name),
        ("signature", _relation_signature(relation)),
        ("inputs", list(relation.inputs)),
        ("outputs", list(relation.outputs)),
        ("rel_tol_default", relation.rel_tol_default),
        ("abs_tol_default", relation.abs_tol_default),
        ("tags", list(relation.tags or ())),
        ("constraints", list(relation.constraints or ())),
        ("soft_constraints", list(relation.soft_constraints or ())),
        ("inverse_targets", sorted(list(relation.inverse_functions))),
    ]
    return _html_section("Relation", relation_items)


def _variable_aliases(spec: dict[str, object]) -> list[str]:
    """Extract variable aliases from one registry specification.

    Args:
        spec: Variable registry entry.

    Returns:
        Flat list of alias strings.
    """
    # Read the aliases field and normalize all supported container types.
    raw_aliases = (spec or {}).get("aliases")
    if raw_aliases is None:
        return []

    alias_values: list[object]
    if isinstance(raw_aliases, dict):
        alias_values = [*raw_aliases.keys(), *raw_aliases.values()]
    elif isinstance(raw_aliases, (list, tuple, set)):
        alias_values = list(raw_aliases)
    else:
        alias_values = [raw_aliases]

    # Keep insertion order while filtering out empty values.
    aliases: list[str] = []
    seen: set[str] = set()
    for value in alias_values:
        alias = str(value).strip()
        normalized = alias.lower()
        if not alias or normalized in seen:
            continue
        aliases.append(alias)
        seen.add(normalized)
    return aliases


def _variable_descriptions(spec: dict[str, object]) -> list[str]:
    """Extract variable descriptions from one registry specification.

    Args:
        spec: Variable registry entry.

    Returns:
        Flat list of description strings.
    """
    # Read the description field and normalize all supported container types.
    raw_descriptions = (spec or {}).get("description")
    if raw_descriptions is None:
        return []

    description_values: list[object]
    if isinstance(raw_descriptions, dict):
        description_values = [*raw_descriptions.keys(), *raw_descriptions.values()]
    elif isinstance(raw_descriptions, (list, tuple, set)):
        description_values = list(raw_descriptions)
    else:
        description_values = [raw_descriptions]

    # Keep insertion order while filtering out empty values.
    descriptions: list[str] = []
    seen: set[str] = set()
    for value in description_values:
        description = str(value).strip()
        normalized = description.lower()
        if not description or normalized in seen:
            continue
        descriptions.append(description)
        seen.add(normalized)
    return descriptions


def _build_relationsystem() -> RelationSystem:
    """Build a RelationSystem instance from registered relations and variable registry.

    Returns:
        Ready-to-use relation system with internal graph metadata.
    """
    # Ensure relation modules are imported before reading the registry list.
    relations.import_relations()
    relation_list = list(_RELATION_REGISTRY)

    # Build variable objects from the allowed-variable registry entries.
    allowed_vars, _, _ = load_allowed_variables()
    variables = []
    for name, spec in allowed_vars.items():
        ndim = int((spec or {}).get("ndim", 0))
        variables.append(
            make_variable(
                name=name,
                ndim=ndim,
                unit=(spec or {}).get("default_unit"),
                method=(spec or {}).get("method"),
                rel_tol=(spec or {}).get("rel_tol"),
                abs_tol=(spec or {}).get("abs_tol"),
                fixed=bool((spec or {}).get("fixed", False)),
            )
        )

    # Use check mode because plotting only needs topology/metadata.
    return RelationSystem(relations=relation_list, variables=variables, mode="check", verbose=False)


def relation_graph_data() -> tuple[list[RelationGraphNode], list[RelationGraphEdge]]:
    """Return variable nodes and conceptual variable-to-variable relation edges.

    Returns:
        Tuple of variable-node and edge summaries.
    """
    relation_system = _build_relationsystem()
    allowed_vars, _, _ = load_allowed_variables()

    # Build variable summaries from the relation-system variable order.
    nodes: list[RelationGraphNode] = []
    for name in relation_system._graph["var_order"]:
        spec = allowed_vars.get(name, {}) or {}
        variable = relation_system._graph["vars"].get(name)
        detail_html = _variable_detail_html(name, spec, variable)
        search_blob = " ".join([name, *[str(value) for value in spec.values()]]).lower()
        nodes.append(RelationGraphNode(name=name, detail_html=detail_html, search_blob=search_blob))

    # Build conceptual edges from relation input/output signatures.
    edges: list[RelationGraphEdge] = []
    for relation in relation_system.relations:
        for output in relation.outputs:
            inputs = relation_input_names(relation, output)
            for source in inputs:
                search_blob = " ".join(
                    [
                        relation.name,
                        output,
                        *inputs,
                        *list(relation.tags or ()),
                        *[str(item) for item in (relation.constraints or ())],
                    ]
                ).lower()
                edges.append(
                    RelationGraphEdge(
                        source=source,
                        target=output,
                        relation=relation.name,
                        color=_color_for(relation.name),
                        detail_html=_relation_detail_html(relation),
                        search_blob=search_blob,
                    )
                )

    return nodes, edges


def relation_graph_plotter(*, width: int = 1200, height: int = 950):
    """Render the RelationSystem internal graph with search and details panels.

    Args:
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        Bokeh layout containing graph, search, and details panels.
    """
    from bokeh.layouts import column, row
    from bokeh.models import (
        AutocompleteInput,
        BoxSelectTool,
        CustomJS,
        Div,
        HoverTool,
        LabelSet,
        MultiLine,
        NodesAndAdjacentNodes,
        PanTool,
        Plot,
        Range1d,
        ResetTool,
        SaveTool,
        Scatter,
        TapTool,
        WheelZoomTool,
    )
    from bokeh.plotting import from_networkx

    relation_system = _build_relationsystem()
    allowed_vars, _, _ = load_allowed_variables()

    # Build a bipartite graph directly from RelationSystem internal adjacency maps.
    graph = nx.Graph()
    relation_to_node_id: dict[object, str] = {}
    variable_node_ids: list[str] = []
    relation_node_ids: list[str] = []
    search_lookup: dict[str, set[str]] = {}
    search_labels: dict[str, str] = {}

    # Add variable nodes first using circle markers.
    for name in relation_system._graph["var_order"]:
        variable = relation_system._graph["vars"].get(name)
        spec = allowed_vars.get(name, {}) or {}
        aliases = _variable_aliases(spec)
        descriptions = _variable_descriptions(spec)
        search_terms = [name, *aliases, *descriptions]
        search_blob = " ".join(term.lower() for term in search_terms)
        variable_node_id = f"var::{name}"
        variable_node_ids.append(variable_node_id)
        graph.add_node(
            variable_node_id,
            node_id=variable_node_id,
            kind="variable",
            name=name,
            label=name,
            marker="circle",
            size=min(30, max(20, 16 + len(name))),
            fill_color=VARIABLE_FILL_COLOR,
            line_color=VARIABLE_LINE_COLOR,
            text_color=VARIABLE_TEXT_COLOR,
            search_blob=search_blob,
            detail_html=_variable_detail_html(name, spec, variable),
        )
        for term in search_terms:
            normalized = term.lower()
            search_lookup.setdefault(normalized, set()).add(variable_node_id)
            search_labels.setdefault(normalized, term)

    # Add relation nodes and keep an id map for edge creation.
    for index, relation in enumerate(relation_system.relations):
        relation_id = f"rel::{index}"
        relation_to_node_id[relation] = relation_id
        relation_node_ids.append(relation_id)
        wrapped_label = _relation_label(relation.name)
        relation_terms = [relation.name, relation.name.replace("_", " ")]
        relation_blob = " ".join(term.lower() for term in relation_terms)
        graph.add_node(
            relation_id,
            node_id=relation_id,
            kind="relation",
            name=relation.name,
            label=wrapped_label,
            marker="square",
            size=min(54, max(28, 26 + 3 * wrapped_label.count("\n") + len(relation.name) // 8)),
            fill_color=RELATION_FILL_COLOR,
            line_color=RELATION_LINE_COLOR,
            text_color=RELATION_TEXT_COLOR,
            search_blob=relation_blob,
            detail_html=_relation_detail_html(relation),
        )
        for term in relation_terms:
            normalized = term.lower()
            search_lookup.setdefault(normalized, set()).add(relation_id)
            search_labels.setdefault(normalized, term)

    # Add relation-defined edges between each relation and its adjacent variables.
    for relation, relation_node_id in relation_to_node_id.items():
        adjacent_variables = relation_system._graph["rels_to_vars"].get(relation, ())
        for variable_name in adjacent_variables:
            variable_node_id = f"var::{variable_name}"
            if variable_node_id not in graph:
                continue
            graph.add_edge(
                relation_node_id,
                variable_node_id,
                relation=relation.name,
                source_label=relation.name,
                target_label=variable_name,
            )

    # Precompute one-hop neighbor lists for the details panel.
    for node_id in graph.nodes:
        adjacent_labels = sorted(str(graph.nodes[neighbor]["name"]) for neighbor in graph.neighbors(node_id))
        if adjacent_labels:
            adjacent_html = "<br>".join(f"- {html.escape(label)}" for label in adjacent_labels)
        else:
            adjacent_html = "<span style='color:#666'>No adjacent nodes.</span>"
        graph.nodes[node_id]["adjacent_html"] = adjacent_html

    # Place variables on a large outer circle.
    graph_layout: dict[str, tuple[float, float]] = {}
    variable_count = len(variable_node_ids)
    variable_radius = max(5.0, 0.12 * variable_count) if variable_count else 5.0
    for index, variable_node_id in enumerate(variable_node_ids):
        angle = -math.pi / 2.0 + (2.0 * math.pi * index) / max(1, variable_count)
        graph_layout[variable_node_id] = (
            variable_radius * math.cos(angle),
            variable_radius * math.sin(angle),
        )

    # Compute relation preferred angles from adjacent variable positions.
    variable_angles = {
        variable_node_id: math.atan2(graph_layout[variable_node_id][1], graph_layout[variable_node_id][0])
        for variable_node_id in variable_node_ids
    }
    relation_count = len(relation_node_ids)
    relation_entries: list[tuple[str, float, str]] = []
    for relation, relation_node_id in relation_to_node_id.items():
        adjacent_variables = relation_system._graph["rels_to_vars"].get(relation, ())
        adjacent_ids = [f"var::{name}" for name in adjacent_variables if f"var::{name}" in variable_angles]
        if adjacent_ids:
            sin_sum = sum(math.sin(variable_angles[node_id]) for node_id in adjacent_ids)
            cos_sum = sum(math.cos(variable_angles[node_id]) for node_id in adjacent_ids)
            preferred_angle = math.atan2(sin_sum, cos_sum) if (sin_sum or cos_sum) else 0.0
        else:
            preferred_angle = -math.pi / 2.0 + (2.0 * math.pi * len(relation_entries)) / max(1, relation_count)
        relation_entries.append((relation_node_id, preferred_angle, relation.name))
    relation_entries.sort(key=lambda item: (item[1], item[2]))

    # Spread relations on inner rings around the center to reduce overlap.
    ring_capacity = 30
    ring_count = max(1, math.ceil(relation_count / ring_capacity))
    if ring_count == 1:
        relation_radii = [variable_radius * 0.55]
    else:
        inner_min = variable_radius * 0.26
        inner_max = variable_radius * 0.76
        radius_step = (inner_max - inner_min) / max(1, ring_count - 1)
        relation_radii = [inner_min + radius_step * ring_index for ring_index in range(ring_count)]

    base_ring_size = relation_count // ring_count
    extra_nodes = relation_count % ring_count
    cursor = 0
    for ring_index, relation_radius in enumerate(relation_radii):
        ring_size = base_ring_size + (1 if ring_index < extra_nodes else 0)
        ring_entries = relation_entries[cursor : cursor + ring_size]
        cursor += ring_size
        if not ring_entries:
            continue

        if ring_size == 1:
            relation_node_id, preferred_angle, _ = ring_entries[0]
            graph_layout[relation_node_id] = (
                relation_radius * math.cos(preferred_angle),
                relation_radius * math.sin(preferred_angle),
            )
            continue

        sin_sum = sum(math.sin(preferred_angle) for _, preferred_angle, _ in ring_entries)
        cos_sum = sum(math.cos(preferred_angle) for _, preferred_angle, _ in ring_entries)
        ring_offset = math.atan2(sin_sum, cos_sum) if (sin_sum or cos_sum) else -math.pi / 2.0
        for ring_pos, (relation_node_id, _, _) in enumerate(ring_entries):
            angle = ring_offset + (2.0 * math.pi * ring_pos) / ring_size
            graph_layout[relation_node_id] = (
                relation_radius * math.cos(angle),
                relation_radius * math.sin(angle),
            )

    # Compute plot ranges around the generated coordinates.
    xs = [xy[0] for xy in graph_layout.values()] if graph_layout else [0.0]
    ys = [xy[1] for xy in graph_layout.values()] if graph_layout else [0.0]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(0.25, x_span * 0.12)
    y_pad = max(0.25, y_span * 0.12)

    plot = Plot(
        width=width,
        height=height,
        x_range=Range1d(min(xs) - x_pad, max(xs) + x_pad),
        y_range=Range1d(min(ys) - y_pad, max(ys) + y_pad),
    )
    plot.title.text = "RelationSystem Graph"
    plot.outline_line_color = "#d9d9d9"

    # Build a Bokeh graph renderer from the precomputed networkx graph.
    graph_renderer = from_networkx(graph, lambda _graph: graph_layout)
    node_source = graph_renderer.node_renderer.data_source
    edge_source = graph_renderer.edge_renderer.data_source
    search_lookup = {key: sorted(value) for key, value in search_lookup.items()}
    search_completions = [search_labels[key] for key in sorted(search_labels, key=lambda item: search_labels[item].lower())]

    # Expose x/y columns for label rendering at node centers.
    node_ids = node_source.data["index"]
    node_source.data["x"] = [float(graph_layout[node_id][0]) for node_id in node_ids]
    node_source.data["y"] = [float(graph_layout[node_id][1]) for node_id in node_ids]

    # Configure node appearance by node type.
    node_glyph = Scatter(
        marker="marker",
        size="size",
        fill_color="fill_color",
        line_color="line_color",
        line_width=1.8,
    )
    graph_renderer.node_renderer.glyph = node_glyph
    graph_renderer.node_renderer.selection_glyph = node_glyph.clone(fill_color=NODE_HIGHLIGHT_COLOR, line_width=2.3)
    graph_renderer.node_renderer.hover_glyph = node_glyph.clone(line_width=2.3)

    # Configure edges to be neutral by default and vivid on highlight.
    edge_glyph = MultiLine(line_color=EDGE_BASE_COLOR, line_alpha=0.65, line_width=1.8)
    graph_renderer.edge_renderer.glyph = edge_glyph
    graph_renderer.edge_renderer.selection_glyph = edge_glyph.clone(
        line_color=EDGE_HIGHLIGHT_COLOR,
        line_alpha=1.0,
        line_width=2.8,
    )
    graph_renderer.edge_renderer.hover_glyph = edge_glyph.clone(line_alpha=0.85, line_width=2.2)

    # Keep one-hop inspection behavior for hover interactions.
    graph_renderer.inspection_policy = NodesAndAdjacentNodes()

    labels = LabelSet(
        x="x",
        y="y",
        text="label",
        source=node_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="7pt",
        text_color="text_color",
    )

    details = Div(
        text=DEFAULT_DETAILS_HTML,
        height=250,
        sizing_mode="stretch_width",
        styles={
            "overflow": "auto",
            "border-left": "1px solid #ddd",
            "padding": "0 0 0 14px",
            "font-size": "13px",
            "line-height": "1.35",
        },
    )
    search_status_default = (
        "<span style='color:#666'>Type a node name, alias, or description keyword and pick a suggestion to jump to that node.</span>"
    )
    search_input = AutocompleteInput(
        title="Search Nodes",
        completions=search_completions,
        case_sensitive=False,
        search_strategy="includes",
        min_characters=1,
        sizing_mode="stretch_width",
        placeholder="name or alias",
    )
    search_status = Div(
        text=search_status_default,
        sizing_mode="stretch_width",
        styles={
            "font-size": "12px",
            "line-height": "1.35",
            "padding": "2px 0 0 0",
            "color": "#555",
        },
    )

    # On node click: highlight node + incident edges + immediate neighbors, then render details.
    node_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                nodeSource=node_source,
                edgeSource=edge_source,
                details=details,
                defaultHtml=DEFAULT_DETAILS_HTML,
            ),
            code="""
if (window.__fusdbNodeSelectionLock) {
  return;
}
window.__fusdbNodeSelectionLock = true;
try {
  if (!cb_obj.indices.length) {
    edgeSource.selected.indices = [];
    details.text = defaultHtml;
    return;
  }

  const rootIndex = cb_obj.indices[0];
  const rootNodeId = nodeSource.data.node_id[rootIndex];

  const activeNodeIds = new Set([rootNodeId]);
  const activeEdgeIndices = [];

  for (let i = 0; i < edgeSource.data.start.length; i++) {
    const start = edgeSource.data.start[i];
    const end = edgeSource.data.end[i];
    if (start !== rootNodeId && end !== rootNodeId) {
      continue;
    }
    activeEdgeIndices.push(i);
    activeNodeIds.add(start);
    activeNodeIds.add(end);
  }

  const activeNodeIndices = [];
  for (let i = 0; i < nodeSource.data.node_id.length; i++) {
    if (activeNodeIds.has(nodeSource.data.node_id[i])) {
      activeNodeIndices.push(i);
    }
  }

  nodeSource.selected.indices = activeNodeIndices;
  edgeSource.selected.indices = activeEdgeIndices;

  const detailHtml = nodeSource.data.detail_html[rootIndex] || defaultHtml;
  const adjacentHtml = nodeSource.data.adjacent_html[rootIndex] || "<span style='color:#666'>No adjacent nodes.</span>";
  details.text = `${detailHtml}<br><br><b>Adjacent Nodes</b><br>${adjacentHtml}`;
} finally {
  window.setTimeout(() => {
    window.__fusdbNodeSelectionLock = false;
  }, 0);
}
	""",
        ),
    )
    # On search submit: locate one node by exact/partial name or alias and trigger selection.
    search_input.js_on_change(
        "value",
        CustomJS(
            args=dict(
                nodeSource=node_source,
                searchLookup=search_lookup,
                searchStatus=search_status,
                searchStatusDefault=search_status_default,
            ),
            code="""
const query = (cb_obj.value || "").trim().toLowerCase();
if (!query) {
  searchStatus.text = searchStatusDefault;
  nodeSource.selected.indices = [];
  return;
}

const exactNodeIds = searchLookup[query] || [];
let candidateNodeIds = [...exactNodeIds];
if (!candidateNodeIds.length) {
  const matches = new Set();
  for (const [term, nodeIds] of Object.entries(searchLookup)) {
    if (!term.includes(query)) {
      continue;
    }
    for (const nodeId of nodeIds) {
      matches.add(nodeId);
    }
  }
  candidateNodeIds = Array.from(matches);
}

if (!candidateNodeIds.length) {
  const searchBlobs = nodeSource.data.search_blob || [];
  for (let i = 0; i < searchBlobs.length; i++) {
    const blob = String(searchBlobs[i] || "").toLowerCase();
    if (!blob.includes(query)) {
      continue;
    }
    candidateNodeIds = [nodeSource.data.node_id[i]];
    break;
  }
}

if (!candidateNodeIds.length) {
  searchStatus.text = "<span style='color:#a33'>No matching node found.</span>";
  nodeSource.selected.indices = [];
  return;
}

if (candidateNodeIds.length > 1 && !exactNodeIds.length) {
  searchStatus.text = `<span style='color:#666'>${candidateNodeIds.length} matches found. Choose a more specific suggestion.</span>`;
  return;
}

const targetNodeId = candidateNodeIds[0];
let targetIndex = -1;
for (let i = 0; i < nodeSource.data.node_id.length; i++) {
  if (nodeSource.data.node_id[i] === targetNodeId) {
    targetIndex = i;
    break;
  }
}
if (targetIndex < 0) {
  searchStatus.text = "<span style='color:#a33'>Match exists but node is unavailable.</span>";
  return;
}

searchStatus.text = `<span style='color:#2f5d50'>Selected: <b>${nodeSource.data.name[targetIndex]}</b></span>`;
nodeSource.selected.indices = [targetIndex];
""",
        ),
    )

    # Add core interaction tools for pan/zoom/select workflows.
    node_hover = HoverTool(
        renderers=[graph_renderer.node_renderer],
        tooltips=[("Node", "@name"), ("Type", "@kind")],
    )
    edge_hover = HoverTool(
        renderers=[graph_renderer.edge_renderer],
        tooltips=[("Relation", "@relation"), ("From", "@source_label"), ("To", "@target_label")],
    )
    tap_tool = TapTool(renderers=[graph_renderer.node_renderer])
    box_tool = BoxSelectTool(renderers=[graph_renderer.node_renderer])
    pan_tool = PanTool()
    zoom_tool = WheelZoomTool()

    plot.add_tools(node_hover, edge_hover, tap_tool, box_tool, pan_tool, zoom_tool, ResetTool(), SaveTool())
    plot.toolbar.active_scroll = zoom_tool
    plot.renderers.extend([graph_renderer, labels])
    search_panel = column(search_input, search_status, width=340)
    inspect_panel = row(search_panel, details, sizing_mode="stretch_width")
    return column(plot, inspect_panel, sizing_mode="stretch_width")


def render_relation_graph_html(*, width: int = 1200, height: int = 950) -> str:
    """Return standalone HTML for the RelationSystem graph layout.

    Args:
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        Standalone HTML document as text.
    """
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    layout = relation_graph_plotter(width=width, height=height)
    return file_html(layout, CDN, "fusdb Relation Graph")


def save_relation_graph_html(path: str | Path, *, width: int = 1200, height: int = 950) -> Path:
    """Write standalone relation-graph HTML to disk.

    Args:
        path: Destination file path.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        Path to the written HTML file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_relation_graph_html(width=width, height=height),
        encoding="utf-8",
    )
    return output_path
