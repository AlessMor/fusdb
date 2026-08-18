from fusdb.plotting.relation_graph import build_relation_graph
from fusdb.registry import RELATIONS


def test_relation_graph_represents_relation_inputs_and_outputs():
    """The public graph builder should expose the dependency graph users inspect."""
    relations = RELATIONS.get_filtered_relations()[:20]
    graph = build_relation_graph(relations)

    assert graph.is_directed()
    for relation in relations:
        relation_node = f"relation::{relation.name}"
        assert relation_node in graph
        for name in relation.input_names:
            assert graph.has_edge(f"variable::{name}", relation_node)
        for name in relation.output_names:
            assert graph.has_edge(relation_node, f"variable::{name}")
