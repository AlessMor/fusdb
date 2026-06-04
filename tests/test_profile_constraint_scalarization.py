"""Tests for profile guard scalarization."""

from __future__ import annotations

from fusdb import Relation, RelationSystem, Variable
from fusdb.relation import constraint_from_expression


def test_outputless_profile_constraint_is_pointwise():
    rel = constraint_from_expression("T_e >= 0", name="T_e_nonnegative")
    system = RelationSystem([Variable("T_e", value=[1.0, 2.0], size=2)], [rel])

    assert system._indices_for_relation(rel) == [0, 1]


def test_profile_to_scalar_relation_is_not_pointwise():
    rel = Relation(
        name="profile reduction",
        func=lambda T_e: 1.0,
        input_names=("T_e",),
        outputs=("p_th",),
    )
    system = RelationSystem(
        [Variable("T_e", value=[1.0, 2.0], size=2), Variable("p_th", value=1.0)],
        [rel],
    )

    assert system._indices_for_relation(rel) == [None]
