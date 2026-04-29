from __future__ import annotations

from fusdb import relation_class as relation_module
from fusdb.relation_class import Relation
from fusdb.relation_class import get_filtered_relations


def test_get_filtered_relations_is_standalone_and_deterministic() -> None:
    """Expected: repeated standalone filtering calls return the same ordered relation names."""
    kwargs = {
        "reactor_tags": ("fusion_power",),
        "variable_methods": (),
    }
    names_1 = [rel.name for rel in get_filtered_relations(**kwargs)]
    names_2 = [rel.name for rel in get_filtered_relations(**kwargs)]

    assert names_1
    assert names_1 == names_2


def test_get_filtered_relations_filters_by_connected_variable_subgraph(monkeypatch) -> None:
    """Expected: variable_names keeps only relations connected to the provided variable seeds."""
    rel_a = Relation.from_callable(
        name="test_a_from_x",
        target="a",
        inputs=("x",),
        func=lambda x: x + 1.0,
    )
    rel_b = Relation.from_callable(
        name="test_b_from_a",
        target="b",
        inputs=("a",),
        func=lambda a: 2.0 * a,
    )
    rel_z = Relation.from_callable(
        name="test_z_from_y",
        target="z",
        inputs=("y",),
        func=lambda y: y + 5.0,
    )

    monkeypatch.setattr(relation_module, "_RELATION_REGISTRY", [])
    filtered = get_filtered_relations(
        reactor_tags=(),
        variable_methods=(),
        variable_names=("x",),
        extra_relations=(rel_a, rel_b, rel_z),
    )
    names = [rel.name for rel in filtered]

    assert names == ["test_a_from_x", "test_b_from_a"]
