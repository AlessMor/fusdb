from pathlib import Path

from fusdb.plotting import __all__ as plotting_exports
from fusdb.plotting.relation_graph import build_relation_graph
from fusdb.registry import RELATIONS


def test_relation_graph_is_the_single_public_graph_builder():
    assert "build_relation_graph" in plotting_exports
    assert "build_relation_node_graph" not in plotting_exports
    assert "build_variable_relation_graph" not in plotting_exports

    relations = RELATIONS.get_filtered_relations()[:20]
    graph = build_relation_graph(relations)
    assert graph.is_directed()
    for relation in relations:
        relation_node = f"relation::{relation.name}"
        assert relation_node in graph
        assert graph.nodes[relation_node]["label"] == relation.name
        assert "description" in graph.nodes[relation_node]
        for name in relation.input_names:
            assert graph.has_edge(f"variable::{name}", relation_node)
        for name in relation.output_names:
            assert graph.has_edge(relation_node, f"variable::{name}")


def test_plotting_has_no_legacy_graph_representations():
    source = Path("src/fusdb/plotting/relation_graph.py").read_text()
    assert "build_relation_node_graph" not in source
    assert "build_variable_relation_graph" not in source
    assert "from itertools import combinations" not in source
    assert source.count("to_undirected(as_view=True)") == 2


def test_relation_system_has_no_duplicate_matching_or_sparsity_graph():
    source = Path("src/fusdb/relationsystem.py").read_text()
    assert "nx.bipartite.maximum_matching" not in source
    assert "_sparsity_dependency_graph" not in source
    assert "_sparsity_graph_cache" not in source
    assert "maximum_bipartite_matching" in source
    assert "_completion_dependency_closure" in source
