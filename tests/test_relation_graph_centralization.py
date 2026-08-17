from itertools import combinations
from pathlib import Path

from fusdb.plotting.relation_graph import (
    build_relation_graph,
    build_relation_node_graph,
    build_variable_relation_graph,
)
from fusdb.registry import RELATIONS


def _edge_sets(graph):
    return {frozenset((source, target)) for source, target in graph.edges}


def test_relation_node_graph_is_projection_of_canonical_graph():
    relations = RELATIONS.get_filtered_relations()[:20]
    directed = build_relation_graph(relations)
    undirected = build_relation_node_graph(relations)
    assert set(directed.nodes) == set(undirected.nodes)
    assert _edge_sets(directed) == _edge_sets(undirected)
    for node, data in directed.nodes(data=True):
        assert data["label"] == node.split("::", 1)[1]
        assert "description" in data


def test_variable_projection_preserves_relation_incidence():
    relations = RELATIONS.get_filtered_relations()[:20]
    graph = build_variable_relation_graph(relations)
    for relation in relations:
        for source, target in combinations(relation.variables, 2):
            assert relation.name in graph[source][target]["relations"]


def test_relation_system_has_no_duplicate_matching_or_sparsity_graph():
    source = Path("src/fusdb/relationsystem.py").read_text()
    assert "nx.bipartite.maximum_matching" not in source
    assert "_sparsity_dependency_graph" not in source
    assert "_sparsity_graph_cache" not in source
    assert "maximum_bipartite_matching" in source
    assert "_completion_dependency_closure" in source
