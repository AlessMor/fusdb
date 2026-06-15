"""Bipartite relation/variable graph of the registry.

Reusable version of ``examples/relation_graph_generator.ipynb``.
"""

from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
from matplotlib.axes import Axes

from fusdb.registry import RELATIONS

from .style import RELATION_COLOR, VARIABLE_COLOR, axes


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
