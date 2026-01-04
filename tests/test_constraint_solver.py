import math

import pytest

from fusdb.relation_class import PRIORITY_STRICT, Relation, RelationSystem


ASPECT_REL = Relation(
    "Aspect ratio",
    ("A", "R", "a"),
    lambda v: v["A"] - v["R"] / v["a"],
    initial_guesses={"A": lambda v: v["R"] / v["a"], "R": lambda v: v["A"] * v["a"], "a": lambda v: v["R"] / v["A"]},
)

EXTENTS_REL = Relation(
    "Major radius",
    ("R", "R_max", "R_min"),
    lambda v: v["R"] - (v["R_max"] + v["R_min"]) / 2,
    initial_guesses={"R": lambda v: (v["R_max"] + v["R_min"]) / 2},
)


def _solve(relations: list[Relation], data: dict[str, float]) -> dict[str, float | None]:
    system = RelationSystem(relations)
    for k, v in data.items():
        system.set(k, v)
    return system.solve()


def test_solver_computes_missing_value_bidirectionally() -> None:
    values = _solve([ASPECT_REL], {"R": 6.0, "A": 3.0})
    assert math.isclose(values["a"] or 0.0, 2.0)


def test_solver_chains_relations_across_equations() -> None:
    values = _solve([EXTENTS_REL, ASPECT_REL], {"R_max": 5.0, "R_min": 3.0, "A": 4.0})
    assert math.isclose(values["R"] or 0.0, 4.0)
    assert math.isclose(values["a"] or 0.0, 1.0)


def test_conflict_respects_explicit_priority_by_default() -> None:
    with pytest.warns(UserWarning):
        values = _solve([ASPECT_REL], {"R": 3.0, "a": 1.0, "A": 2.5})
    assert math.isclose(values["A"] or 0.0, 2.5)


def test_strict_relation_priority_can_override_explicit_values() -> None:
    rel = ASPECT_REL.with_tol(ASPECT_REL.rel_tol, priority=PRIORITY_STRICT)
    with pytest.warns(UserWarning):
        values = _solve([rel], {"R": 3.0, "a": 1.0, "A": 2.5})
    assert math.isclose(values["A"] or 0.0, 3.0)


def test_violation_warning_reports_expected_once() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], lock_explicit=True, warn=lambda msg, _cat=None: warnings.append(msg))
    system.set("R", 3.0)
    system.set("a", 1.0)
    system.set("A", 2.5)

    system.solve()

    assert warnings == ["A violates Aspect ratio relation: expected value is 3.0, got 2.5 (holding R=3.0, a=1.0)"]


def test_lock_explicit_respects_scaled_tolerance() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], rel_tol=1e-2, lock_explicit=True, warn=lambda msg, _cat=None: warnings.append(msg))
    system.set("R", 2.920353982300885)
    system.set("a", 1.0)
    system.set("A", 2.92)  # small residual but well within rel_tol * scale

    system.solve()

    assert warnings == []


def test_lock_explicit_warns_above_scaled_tolerance() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], rel_tol=1e-2, lock_explicit=True, warn=lambda msg, _cat=None: warnings.append(msg))
    system.set("R", 3.0)
    system.set("a", 1.0)
    system.set("A", 3.2)  # ~6.7% high; should warn

    system.solve()

    assert warnings == ["A violates Aspect ratio relation: expected value is 3.0, got 3.2 (holding R=3.0, a=1.0)"]
