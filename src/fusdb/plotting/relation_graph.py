"""Relation graph visualizations from the registry.

Reusable version of ``examples/relation_graph_generator.ipynb``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import networkx as nx
from matplotlib.axes import Axes

from fusdb.registry import RELATIONS, VARIABLES

from .style import RELATION_COLOR, VARIABLE_COLOR, axes


def _short_label(label: str, *, limit: int = 6) -> str:
    """Return a compact node label for drawing inside a glyph."""
    return label if len(label) <= limit else f"{label[:limit]}..."


def build_relation_graph(relations: Iterable[Any] | None = None) -> nx.DiGraph:
    """Build a directed relation -> variable graph.

    Each relation becomes a ``kind="relation"`` node; its inputs point into it
    and its outputs point out of it (both ``kind="variable"`` nodes).

    Args:
        relations: Relations to include. Defaults to the registry's filtered
            relations (one producer per output where ``default_relation`` is set).

    Returns:
        A :class:`networkx.DiGraph` with ``kind`` and ``label`` node attributes.
    """
    # TODO:remove in favor of building a graph from RelationSystem
    if relations is None:
        relations = RELATIONS.get_filtered_relations()

    graph = nx.DiGraph()
    for relation in relations:
        relation_node = f"relation::{relation.name}"
        graph.add_node(relation_node, kind="relation", label=relation.name)
        for name in relation.input_names:
            graph.add_node(f"variable::{name}", kind="variable", label=name)
            graph.add_edge(f"variable::{name}", relation_node)
        for name in relation.outputs:
            graph.add_node(f"variable::{name}", kind="variable", label=name)
            graph.add_edge(relation_node, f"variable::{name}")
    return graph


def build_variable_relation_graph(relations: Iterable[Any] | None = None) -> nx.Graph:
    """Build an undirected variable graph with relations stored on edges.

    Relations in FusDB are acausal, so this projection treats every variable
    touched by a relation as mutually connected. A relation touching more than
    two variables becomes pairwise edges among those variables; multiple
    relations on the same variable pair are aggregated on one edge.
    """
    if relations is None:
        relations = RELATIONS.get_filtered_relations()

    graph = nx.Graph()
    for relation in relations:
        variable_names = tuple(relation.variables)
        for name in variable_names:
            spec = VARIABLES.get(name)
            graph.add_node(
                name,
                kind="variable",
                label=name,
                aliases=", ".join(spec.aliases),
                unit=spec.unit,
                description=spec.description,
            )

        for source, target in combinations(variable_names, 2):
            if graph.has_edge(source, target):
                data = graph[source][target]
                data["relations"].append(relation.name)
                data["relation_tags"].extend(
                    tag for tag in relation.tags if tag not in data["relation_tags"]
                )
            else:
                graph.add_edge(
                    source,
                    target,
                    relations=[relation.name],
                    relation_tags=list(relation.tags),
                )

    for _, _, data in graph.edges(data=True):
        data["relation_count"] = len(data["relations"])
        data["label"] = ", ".join(data["relations"])
        data["tags"] = ", ".join(data["relation_tags"])

    return graph


def build_relation_node_graph(relations: Iterable[Any] | None = None) -> nx.Graph:
    """Build an undirected graph with both variables and relations as nodes."""
    if relations is None:
        relations = RELATIONS.get_filtered_relations()

    graph = nx.Graph()
    for relation in relations:
        relation_node = f"relation::{relation.name}"
        graph.add_node(
            relation_node,
            kind="relation",
            label=relation.name,
            aliases="",
            unit="",
            tags=", ".join(relation.tags),
            description=relation.source_name,
        )
        for name in relation.variables:
            spec = VARIABLES.get(name)
            variable_node = f"variable::{name}"
            graph.add_node(
                variable_node,
                kind="variable",
                label=name,
                aliases=", ".join(spec.aliases),
                unit=spec.unit,
                tags="",
                description=spec.description,
            )
            graph.add_edge(relation_node, variable_node)

    return graph


def plot_relation_graph(
    graph: nx.DiGraph | None = None,
    *,
    ax: Axes | None = None,
    seed: int = 7,
    k: float = 0.24,
    labels: bool = False,
) -> Axes:
    """Draw the relation/variable graph with a spring layout.

    Args:
        graph: Graph to draw; built from the registry when omitted.
        ax: Existing axis to draw on; a new figure is created when omitted.
        seed: Spring-layout random seed for reproducible positions.
        k: Spring-layout optimal node distance (smaller packs nodes tighter).
        labels: When ``True``, annotate every node with its label.

    Returns:
        The axis the graph was drawn on.
    """
    if graph is None:
        graph = build_relation_graph()

    ax = axes(ax, figsize=(16, 10))
    positions = nx.spring_layout(graph, seed=seed, k=k)
    relation_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "relation"]
    variable_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "variable"]

    nx.draw_networkx_nodes(
        graph, positions, nodelist=relation_nodes, node_size=90,
        node_color=RELATION_COLOR, alpha=0.85, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=variable_nodes, node_size=40,
        node_color=VARIABLE_COLOR, alpha=0.70, ax=ax,
    )
    nx.draw_networkx_edges(graph, positions, width=0.35, alpha=0.18, arrows=False, ax=ax)
    if labels:
        node_labels = {node: data["label"] for node, data in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, positions, labels=node_labels, font_size=6, ax=ax)

    ax.set_axis_off()
    return ax


def bokeh_relation_graph(
    graph: nx.Graph | None = None,
    *,
    seed: int = 7,
    k: float = 0.42,
    width: int = 1300,
    height: int = 900,
    labels: bool = False,
    title: str = "Relation-variable graph from current registries",
) -> Any:
    """Build an interactive Bokeh relation/variable graph.

    Variables and acausal relations are both nodes. Each node shows a compact
    label; hover exposes the full name and metadata.
    """
    if graph is None:
        graph = build_relation_node_graph()

    from bokeh.layouts import column
    from bokeh.models import (
        AutocompleteInput,
        ColumnDataSource,
        CustomJS,
        HoverTool,
        LabelSet,
        Range1d,
    )
    from bokeh.plotting import figure

    positions = nx.spring_layout(graph, seed=seed, k=k)
    if positions:
        x_values = [float(x) for x, _ in positions.values()]
        y_values = [float(y) for _, y in positions.values()]
        x_pad = max((max(x_values) - min(x_values)) * 0.08, 0.1)
        y_pad = max((max(y_values) - min(y_values)) * 0.08, 0.1)
        x_range = Range1d(min(x_values) - x_pad, max(x_values) + x_pad)
        y_range = Range1d(min(y_values) - y_pad, max(y_values) + y_pad)
    else:
        x_range = Range1d(-1, 1)
        y_range = Range1d(-1, 1)

    node_rows = []
    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        kind = data["kind"]
        is_relation = kind == "relation"
        node_rows.append(
            {
                "x": float(x),
                "y": float(y),
                "label": data["label"],
                "short_label": _short_label(data["label"]),
                "kind": kind,
                "node_id": node,
                "aliases": data.get("aliases", ""),
                "unit": data.get("unit", ""),
                "tags": data.get("tags", ""),
                "color": RELATION_COLOR if is_relation else VARIABLE_COLOR,
                "base_color": RELATION_COLOR if is_relation else VARIABLE_COLOR,
                "size": 54 if is_relation else 44,
                "base_size": 54 if is_relation else 44,
                "alpha": 0.92 if is_relation else 0.86,
                "base_alpha": 0.92 if is_relation else 0.86,
                "search_blob": " ".join(
                    str(item).lower()
                    for item in (
                        data["label"],
                        data.get("aliases", ""),
                        data.get("unit", ""),
                        data.get("tags", ""),
                        data.get("description", ""),
                    )
                ),
            }
        )

    edge_rows = []
    for source, target, data in graph.edges(data=True):
        x_source, y_source = positions[source]
        x_target, y_target = positions[target]
        edge_rows.append(
            {
                "xs": [float(x_source), float(x_target)],
                "ys": [float(y_source), float(y_target)],
                "source_id": source,
                "target_id": target,
                "source": graph.nodes[source]["label"],
                "target": graph.nodes[target]["label"],
                "alpha": 0.22,
                "width": 0.7,
                "search_blob": " ".join(
                    str(item).lower()
                    for item in (
                        graph.nodes[source]["label"],
                        graph.nodes[target]["label"],
                    )
                ),
            }
        )

    node_source = ColumnDataSource(
        {
            key: [row[key] for row in node_rows]
            for key in (
                "x",
                "y",
                "label",
                "short_label",
                "kind",
                "node_id",
                "aliases",
                "unit",
                "tags",
                "color",
                "base_color",
                "size",
                "base_size",
                "alpha",
                "base_alpha",
                "search_blob",
            )
        }
    )
    edge_source = ColumnDataSource(
        {
            key: [row[key] for row in edge_rows]
            for key in (
                "xs",
                "ys",
                "source_id",
                "target_id",
                "source",
                "target",
                "alpha",
                "width",
                "search_blob",
            )
        }
    )
    completions = sorted(
        {
            item
            for row in node_rows
            for item in (
                row["label"],
                *(alias.strip() for alias in row["aliases"].split(",") if alias.strip()),
                *(tag.strip() for tag in row["tags"].split(",") if tag.strip()),
            )
        },
        key=str.lower,
    )

    plot = figure(
        title=title,
        width=width,
        height=height,
        x_range=x_range,
        y_range=y_range,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        toolbar_location="above",
    )
    plot.grid.visible = False
    plot.axis.visible = False
    plot.outline_line_color = "#d6d6d6"

    plot.multi_line(
        "xs",
        "ys",
        source=edge_source,
        line_width="width",
        line_alpha="alpha",
        line_color="#6f6f6f",
    )
    nodes = plot.scatter(
        "x",
        "y",
        source=node_source,
        size="size",
        fill_color="color",
        line_color="#ffffff",
        line_width=0.5,
        fill_alpha="alpha",
        legend_group="kind",
    )
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="short_label",
            source=node_source,
            text_font_size="7pt",
            text_color="#ffffff",
            text_align="center",
            text_baseline="middle",
            text_alpha=0.96,
        )
    )
    plot.add_tools(
        HoverTool(
            renderers=[nodes],
            tooltips=[
                ("name", "@label"),
                ("kind", "@kind"),
                ("aliases", "@aliases"),
                ("unit", "@unit"),
                ("tags", "@tags"),
            ],
        )
    )
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"

    if labels and node_rows:
        plot.add_layout(
            LabelSet(
                x="x",
                y="y",
                text="label",
                source=node_source,
                text_font_size="8pt",
                x_offset=5,
                y_offset=5,
                text_alpha=0.82,
            )
        )

    search = AutocompleteInput(
        title="Search variables, aliases, relations, or tags",
        completions=completions,
        case_sensitive=False,
        max_completions=20,
        width=width,
    )
    search.js_on_change(
        "value",
        CustomJS(
            args={"node_source": node_source, "edge_source": edge_source, "plot": plot},
            code="""
const query = cb_obj.value.trim().toLowerCase();
const nodes = node_source.data;
const edges = edge_source.data;
const matchedNodes = new Set();
const relatedNodes = new Set();
let firstNode = -1;

for (let i = 0; i < nodes.label.length; i++) {
  const matched = query !== "" && nodes.search_blob[i].includes(query);
  if (matched) {
    matchedNodes.add(nodes.node_id[i]);
    if (firstNode < 0) {
      firstNode = i;
    }
  }
}

for (let i = 0; i < edges.source.length; i++) {
  const edgeMatched = query !== "" && edges.search_blob[i].includes(query);
  const endpointMatched = matchedNodes.has(edges.source_id[i]) || matchedNodes.has(edges.target_id[i]);
  if (edgeMatched || endpointMatched) {
    relatedNodes.add(edges.source_id[i]);
    relatedNodes.add(edges.target_id[i]);
    edges.alpha[i] = 0.82;
    edges.width[i] = edgeMatched ? 2.8 : 1.8;
  } else {
    edges.alpha[i] = query === "" ? 0.22 : 0.035;
    edges.width[i] = query === "" ? 0.7 : 0.4;
  }
}

for (let i = 0; i < nodes.label.length; i++) {
  if (query === "") {
    nodes.color[i] = nodes.base_color[i];
    nodes.size[i] = nodes.base_size[i];
    nodes.alpha[i] = nodes.base_alpha[i];
  } else if (matchedNodes.has(nodes.node_id[i])) {
    nodes.color[i] = "#e7298a";
    nodes.size[i] = Math.max(nodes.base_size[i] + 5, 12);
    nodes.alpha[i] = 1.0;
  } else if (relatedNodes.has(nodes.node_id[i])) {
    nodes.color[i] = "#66a61e";
    nodes.size[i] = Math.max(nodes.base_size[i] + 2, 9);
    nodes.alpha[i] = 0.92;
  } else {
    nodes.color[i] = nodes.base_color[i];
    nodes.size[i] = nodes.base_size[i];
    nodes.alpha[i] = 0.16;
  }
}

if (firstNode >= 0) {
  const spanX = plot.x_range.end - plot.x_range.start;
  const spanY = plot.y_range.end - plot.y_range.start;
  const x = nodes.x[firstNode];
  const y = nodes.y[firstNode];
  plot.x_range.start = x - spanX / 2;
  plot.x_range.end = x + spanX / 2;
  plot.y_range.start = y - spanY / 2;
  plot.y_range.end = y + spanY / 2;
}

node_source.change.emit();
edge_source.change.emit();
""",
        ),
    )

    return column(search, plot)


def relation_graph_html(
    graph: nx.Graph | None = None,
    *,
    title: str = "Relation-variable graph from current registries",
    **kwargs: Any,
) -> str:
    """Render the interactive relation graph as standalone Bokeh HTML."""
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(bokeh_relation_graph(graph, title=title, **kwargs), CDN, title)
