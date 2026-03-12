import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relation_class import Relation
from fusdb.relation_util import _RELATION_REGISTRY, relation, try_sympify_expression


def _sum_relation(a: float, b: float) -> float:
    return a + b


def test_relation_evaluate_and_variables():
    """Expected: canonical variables/aliases are exposed and evaluation works for default and explicit targets."""
    rel = Relation.from_callable(
        name="sum",
        target="c",
        func=_sum_relation,
        inputs=("a", "b"),
        numeric_functions={"a": (("b", "c"), lambda b, c: c - b)},
    )
    assert tuple(rel.variables) == ("a", "b", "c")
    assert rel.preferred_target == "c"
    assert rel.required_inputs() == ("a", "b")
    assert rel.evaluate({"a": 1.0, "b": 2.0}) == pytest.approx(3.0)
    assert rel.evaluate({"b": 2.0, "c": 5.0}, target="a") == pytest.approx(3.0)


def test_relation_canonicalizes_variable_names():
    """Expected: variable names are canonicalized and the preferred target resolves to canonical output."""
    rel = Relation.from_callable(
        name="sum",
        func=_sum_relation,
        target="aspect_ratio",
        inputs=("major_radius", "minor_radius"),
        variables=("major_radius", "minor_radius", "aspect_ratio"),
    )
    assert tuple(rel.variables) == ("R", "a", "A")
    assert rel.required_inputs() == ("R", "a")
    assert (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None)) == "A"


def test_relation_solve_for_value_uses_inverse_function_override():
    """Expected: solve_for_value uses an explicit inverse override when one is provided."""
    rel = Relation.from_callable(
        name="sum",
        func=_sum_relation,
        target="c",
        variables={"a": None, "b": None, "c": None},
        inputs=("a", "b"),
        inverse_functions={"a": lambda values: values["c"] - values["b"]},
    )
    assert rel.solve_for_value("a", {"b": 3.0, "c": 8.0}) == pytest.approx(5.0)


def test_relation_symbolic_inverse_solver():
    """Expected: symbolic inverse is cached and solve_for_value resolves invertible unknowns, returning None for unknown vars."""
    rel = Relation.from_callable(
        name="sum",
        func=_sum_relation,
        target="c",
        inputs=("a", "b"),
    )

    solver = rel.inverse_solver("a")
    solver_cached = rel.inverse_solver("a")
    assert solver is not None
    assert solver_cached is solver
    assert float(solver(3.0, 8.0)) == pytest.approx(5.0)
    assert rel.solve_for_value("a", {"b": 3.0, "c": 8.0}) == pytest.approx(5.0)
    assert rel.solve_for_value("b", {"a": 3.0, "c": 8.0}) == pytest.approx(5.0)
    assert rel.solve_for_value("c", {"a": 3.0, "b": 5.0}) == pytest.approx(8.0)
    assert rel.solve_for_value("x", {"b": 3.0, "c": 8.0}) is None


def test_relation_symbolic_build_warns_when_unsympifiable():
    """Expected: non-sympifiable callables emit a warning and produce no symbolic expression."""
    def _bad(a: float) -> object:
        return {"bad": a}

    with pytest.warns(RuntimeWarning, match="Could not convert relation"):
        rel = Relation.from_callable(
            name="bad_symbolic",
            func=_bad,
            target="b",
            inputs=("a",),
        )
    assert rel.sympy_expression is None


def test_try_sympify_expression_warns_on_invalid_string():
    """Expected: invalid symbolic strings emit a warning and return None."""
    with pytest.warns(RuntimeWarning, match="could not parse expression"):
        parsed = try_sympify_expression("a + )", context="test")
    assert parsed is None


def test_relation_decorator_in_relation_util_registers_relation():
    """Expected: @relation returns a Relation object and appends it to the global registry."""
    before = len(_RELATION_REGISTRY)

    @relation(name="sum_decorated", output="c", tags=("test",))
    def sum_decorated(a: float, b: float) -> float:
        return a + b

    assert isinstance(sum_decorated, Relation)
    assert sum_decorated.evaluate({"a": 2.0, "b": 5.0}) == pytest.approx(7.0)
    assert len(_RELATION_REGISTRY) == before + 1
    assert _RELATION_REGISTRY[-1] is sum_decorated
    _RELATION_REGISTRY.pop()


def test_symbolic_proxy_preserves_keyword_only_defaults():
    """Expected: symbolic model construction handles keyword-only defaults and yields a symbolic expression."""
    def _kw_pick(*args, _i=1) -> object:
        return np.sqrt(args[_i] ** 2)

    rel = Relation.from_callable(
        name="kw_proxy",
        func=_kw_pick,
        target="c",
        inputs=("a", "b"),
    )
    assert rel.sympy_expression is not None
