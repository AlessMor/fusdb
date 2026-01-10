import math

import pytest

from fusdb.relation_class import Relation, RelationSystem
from fusdb.relation_util import symbol


ASPECT_REL = Relation(
    "Aspect ratio",
    ("A", "R", "a"),
    symbol("A") - symbol("R") / symbol("a"),
)

EXTENTS_REL = Relation(
    "Major radius",
    ("R", "R_max", "R_min"),
    symbol("R") - (symbol("R_max") + symbol("R_min")) / 2,
)


def _solve(relations: list[Relation], data: dict[str, float]) -> dict[str, object]:
    system = RelationSystem(relations)
    for k, v in data.items():
        system.set(k, v)
    return system.solve()


def test_solver_computes_missing_value_bidirectionally() -> None:
    values = _solve([ASPECT_REL], {"R": 6.0, "A": 3.0})
    assert math.isclose(float(values["a"]), 2.0)


def test_solver_chains_relations_across_equations() -> None:
    values = _solve([EXTENTS_REL, ASPECT_REL], {"R_max": 5.0, "R_min": 3.0, "A": 4.0})
    assert math.isclose(float(values["R"]), 4.0)
    assert math.isclose(float(values["a"]), 1.0)


def test_inconsistent_explicit_warns() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], warn=lambda msg, _cat=None: warnings.append(msg))
    system.set("R", 3.0)
    system.set("a", 1.0)
    system.set("A", 2.5)

    values = system.solve()

    assert math.isclose(float(values["A"]), 2.5)
    assert warnings == ["A violates Aspect ratio relation: expected value is 3.0, got 2.5 (holding R=3.0, a=1.0)"]


def test_explicit_within_tolerance_does_not_warn() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], rel_tol=1e-2, warn=lambda msg, _cat=None: warnings.append(msg))
    system.set("R", 2.920353982300885)
    system.set("a", 1.0)
    system.set("A", 2.92)

    system.solve()

    assert warnings == []


def test_explicit_above_tolerance_warns() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], rel_tol=1e-2, warn=lambda msg, _cat=None: warnings.append(msg))
    system.set("R", 3.0)
    system.set("a", 1.0)
    system.set("A", 3.2)

    system.solve()

    assert warnings == ["A violates Aspect ratio relation: expected value is 3.0, got 3.2 (holding R=3.0, a=1.0)"]
