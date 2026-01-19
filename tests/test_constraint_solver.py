import math

from fusdb.relation_class import Relation, RelationSystem
from fusdb.relation_util import symbol


ASPECT_REL = Relation(
    "Aspect ratio",
    ("A", "R", "a"),
    symbol("A") - symbol("R") / symbol("a"),
)

ASPECT_REL_SOLVE_R = Relation(
    "Aspect ratio (solve R)",
    ("A", "R", "a"),
    symbol("A") - symbol("R") / symbol("a"),
    solve_for=("R",),
)

EXTENTS_REL = Relation(
    "Major radius",
    ("R", "R_max", "R_min"),
    symbol("R") - (symbol("R_max") + symbol("R_min")) / 2,
)

BLOCK_REL_1 = Relation(
    "Block eq 1",
    ("x", "y"),
    symbol("x") + symbol("y") - 3,
)

BLOCK_REL_2 = Relation(
    "Block eq 2",
    ("x", "y"),
    symbol("x") - symbol("y") - 1,
)


def _solve(
    relations: list[Relation],
    data: dict[str, float],
    *,
    rel_tol: float | None = None,
    mode: str = "override_input",
) -> dict[str, object]:
    kwargs = {}
    if rel_tol is not None:
        kwargs["rel_tol"] = rel_tol
    system = RelationSystem(relations, **kwargs)
    return system.solve(data, mode=mode)


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
    values = system.solve({"R": 3.0, "a": 1.0, "A": 2.5})

    assert math.isclose(float(values["A"]), 3.0)
    assert any("Explicit A overridden" in warning for warning in warnings)


def test_explicit_within_tolerance_does_not_warn() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], rel_tol=1e-2, warn=lambda msg, _cat=None: warnings.append(msg))
    system.solve({"R": 2.920353982300885, "a": 1.0, "A": 2.92})

    assert warnings == []


def test_explicit_above_tolerance_warns() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL], rel_tol=1e-2, warn=lambda msg, _cat=None: warnings.append(msg))
    values = system.solve({"R": 3.0, "a": 1.0, "A": 3.2})

    assert math.isclose(float(values["A"]), 3.0)
    assert any("Explicit A overridden" in warning for warning in warnings)


def test_explicit_input_override_warns() -> None:
    warnings: list[str] = []
    system = RelationSystem([ASPECT_REL_SOLVE_R], warn=lambda msg, _cat=None: warnings.append(msg))
    values = system.solve({"R": 2.0, "a": 1.0, "A": 3.0})

    assert math.isclose(float(values["R"]), 3.0)
    assert any("Explicit R overridden" in warning for warning in warnings)


def test_block_solves_without_explicit_values() -> None:
    values = _solve([BLOCK_REL_1, BLOCK_REL_2], {})
    assert math.isclose(float(values["x"]), 2.0)
    assert math.isclose(float(values["y"]), 1.0)


def test_override_disabled_raises_with_explicit_blame() -> None:
    system = RelationSystem([BLOCK_REL_1, BLOCK_REL_2])

    try:
        system.solve({"x": 0.0, "y": 0.0}, mode="locked_input")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError for fixed-input inconsistency")

    assert "Unable to satisfy relations" in message
    assert "explicit inputs: x=0.0" in message
