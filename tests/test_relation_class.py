import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relation_class import Relation
from fusdb.relation_util import _RELATION_REGISTRY, relation, relation_input_names, try_sympify_expression
from fusdb.relationsystem_class import RelationSystem
from fusdb.variable_util import make_variable


def _sum_relation(a: float, b: float) -> float:
    return a + b


def _bundle_relation(a: float) -> dict[str, float]:
    return {"b": a + 1.0, "c": 2.0 * a}


def test_relation_evaluate_and_variables():
    """Expected: canonical variables/aliases are exposed and evaluation works for default and explicit targets."""
    rel = Relation.from_callable(
        name="sum",
        target="c",
        func=_sum_relation,
        inputs=("a", "b"),
        solvers={"a": (("b", "c"), lambda b, c: c - b)},
    )
    assert tuple(rel.symbols) == ("a", "b", "c")
    assert rel.outputs == ("c",)
    assert relation_input_names(rel) == ("a", "b")
    assert rel.evaluate({"a": 1.0, "b": 2.0}) == pytest.approx(3.0)
    assert rel.evaluate({"b": 2.0, "c": 5.0}, target="a") == pytest.approx(3.0)


def test_relation_canonicalizes_variable_names():
    """Expected: variable names are canonicalized and the preferred target resolves to canonical output."""
    rel = Relation.from_callable(
        name="sum",
        func=_sum_relation,
        target="aspect_ratio",
        inputs=("major_radius", "minor_radius"),
        symbols=("major_radius", "minor_radius", "aspect_ratio"),
    )
    assert tuple(rel.symbols) == ("R", "a", "A")
    assert relation_input_names(rel) == ("R", "a")
    assert rel.outputs == ("A",)


def test_relation_solve_for_value_uses_inverse_function_override():
    """Expected: solve_for_value uses an explicit inverse override when one is provided."""
    rel = Relation.from_callable(
        name="sum",
        func=_sum_relation,
        target="c",
        symbols={"a": None, "b": None, "c": None},
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


def test_multi_output_relation_is_forward_only():
    """Expected: multi-output relations expose bundle outputs and disable inverse solving."""
    rel = Relation.from_callable(
        name="bundle",
        func=_bundle_relation,
        outputs=("b", "c"),
        inputs=("a",),
    )
    assert rel.outputs == ("b", "c")
    assert rel.is_multi_output is True
    assert rel.is_forward_only is True
    assert relation_input_names(rel) == ("a",)
    applied = rel.apply({"a": 3.0})
    assert applied["b"] == pytest.approx(4.0)
    assert applied["c"] == pytest.approx(6.0)
    evaluated = rel.evaluate({"a": 3.0})
    assert evaluated["b"] == pytest.approx(4.0)
    assert evaluated["c"] == pytest.approx(6.0)
    assert rel.solve_for_value("a", {"b": 4.0, "c": 6.0}) is None


def test_relation_decorator_supports_multiple_outputs():
    """Expected: @relation(outputs=...) registers a forward-only multi-output relation."""
    before = len(_RELATION_REGISTRY)

    @relation(name="bundle_decorated", outputs=("b", "c"), tags=("test",))
    def bundle_decorated(a: float) -> dict[str, float]:
        return {"b": a + 1.0, "c": 2.0 * a}

    assert isinstance(bundle_decorated, Relation)
    applied = bundle_decorated.apply({"a": 2.0})
    assert applied["b"] == pytest.approx(3.0)
    assert applied["c"] == pytest.approx(4.0)
    assert bundle_decorated.is_forward_only is True
    assert len(_RELATION_REGISTRY) == before + 1
    assert _RELATION_REGISTRY[-1] is bundle_decorated
    _RELATION_REGISTRY.pop()


def test_relationsystem_solves_multi_output_relation_atomically():
    """Expected: RelationSystem applies one forward-only relation to multiple outputs in one pass."""
    rel = Relation.from_callable(
        name="bundle",
        func=_bundle_relation,
        outputs=("b", "c"),
        inputs=("a",),
    )
    a = make_variable(name="a", ndim=0)
    a.add_value(3.0, as_input=True)
    b = make_variable(name="b", ndim=0)
    c = make_variable(name="c", ndim=0)
    system = RelationSystem([rel], [a, b, c], mode="overwrite")
    system.solve()
    assert system.variables_dict["b"].current_value == pytest.approx(4.0)
    assert system.variables_dict["c"].current_value == pytest.approx(6.0)


def test_relationsystem_evaluate_handles_multi_output_relation():
    """Expected: RelationSystem.evaluate returns every output produced by a multi-output relation."""
    rel = Relation.from_callable(
        name="bundle",
        func=_bundle_relation,
        outputs=("b", "c"),
        inputs=("a",),
    )
    a = make_variable(name="a", ndim=0)
    b = make_variable(name="b", ndim=0)
    c = make_variable(name="c", ndim=0)
    system = RelationSystem([rel], [a, b, c], mode="overwrite")
    values = system.evaluate({"a": np.asarray([1.0, 2.0, 3.0])})
    assert np.allclose(values["b"], np.asarray([2.0, 3.0, 4.0]))
    assert np.allclose(values["c"], np.asarray([2.0, 4.0, 6.0]))


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
