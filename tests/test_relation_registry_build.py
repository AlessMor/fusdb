"""Build-time validation of the relation registry.

The relation registry canonicalizes every relation against the variable
registry when it is built, rejecting alias-degenerate relations (a declared
output that resolves to one of the relation's own inputs).  These tests pin
that contract so the mistake is caught here -- at registry build -- instead of
being silently dropped when a RelationSystem is later compiled.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusdb.registry import RELATIONS, RelationRegistry
from fusdb.relation import Relation, canonicalize_relation
from fusdb.relationsystem import RelationSystem
from fusdb.variable import Variable


def test_shipped_relations_build_without_degenerate_relations():
    """All shipped relations canonicalize cleanly against the variable registry.

    This fails the moment a newly added relation declares an output that is an
    alias of one of its own inputs.
    """
    registry = RelationRegistry.discover()
    assert len(registry) > 0
    # The lazy global proxy must build to the same set.
    assert len(RELATIONS) == len(registry)


class _Spec:
    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name


class _AliasRegistry:
    """Minimal variable registry mapping names to canonical names."""

    def __init__(self, aliases: dict[str, str]) -> None:
        self._aliases = aliases

    def get(self, name: str) -> _Spec:
        return _Spec(self._aliases.get(name, name))


def _degenerate_relation() -> Relation:
    # Output ``y`` is an alias of input ``x``: after canonicalization both sides
    # are ``x``, so the relation determines nothing.
    return Relation(
        name="degenerate",
        func=lambda x: x,
        input_names=("x",),
        outputs=("y",),
    )


def test_canonicalize_relation_rejects_alias_degenerate():
    registry = _AliasRegistry({"x": "x", "y": "x"})
    with pytest.raises(ValueError, match="alias-degenerate"):
        canonicalize_relation(_degenerate_relation(), registry)


def test_canonicalize_relation_keeps_non_degenerate():
    registry = _AliasRegistry({"x": "x", "y": "y"})
    resolved = canonicalize_relation(_degenerate_relation(), registry)
    assert resolved.input_names == ("x",)
    assert resolved.outputs == ("y",)


def test_registry_build_rejects_alias_degenerate_relation():
    registry = _AliasRegistry({"x": "x", "y": "x"})
    with pytest.raises(ValueError, match="alias-degenerate"):
        RelationRegistry([_degenerate_relation()], variable_registry=registry)


# ── Dual identification (name + function name) and identifier uniqueness ──────

_PASSTHROUGH_REGISTRY = _AliasRegistry({})


def _named_relation(name: str, function_name: str) -> Relation:
    """A trivial ``y = x`` relation with explicit name and function name."""
    return Relation(
        name=name,
        func=lambda x: x,
        input_names=("x",),
        outputs=("y",),
        function_name=function_name,
        argument_names=("x",),
    )


def test_registry_rejects_duplicate_name():
    rels = [_named_relation("A", "f1"), _named_relation("A", "f2")]
    with pytest.raises(ValueError, match="must be unique"):
        RelationRegistry(rels, variable_registry=_PASSTHROUGH_REGISTRY)


def test_registry_rejects_duplicate_function_name():
    rels = [_named_relation("A", "f1"), _named_relation("B", "f1")]
    with pytest.raises(ValueError, match="must be unique"):
        RelationRegistry(rels, variable_registry=_PASSTHROUGH_REGISTRY)


def test_registry_rejects_name_function_cross_collision():
    # B's user-facing name equals A's function name: addressing "f1" would be
    # ambiguous, so it must be rejected even though no name and no function name
    # is duplicated on its own.
    rels = [_named_relation("A", "f1"), _named_relation("f1", "f2")]
    with pytest.raises(ValueError, match="must be unique"):
        RelationRegistry(rels, variable_registry=_PASSTHROUGH_REGISTRY)


def test_registry_allows_name_equal_to_own_function_name():
    # A relation decorated without an explicit name has name == function_name;
    # that is one owner registering both identifiers, not a collision.
    registry = RelationRegistry([_named_relation("same", "same")], variable_registry=_PASSTHROUGH_REGISTRY)
    assert len(registry) == 1


def test_registry_resolves_and_contains_by_name_and_function():
    reg = RelationRegistry.discover()
    name = "Parabolic electron temperature profile"
    function = "parabolic_electron_temperature_profile"
    assert reg.get(name) is reg.get(function)
    assert name in reg
    assert function in reg
    assert "definitely-not-a-relation" not in reg


def test_get_filtered_relations_exclude_and_order_accept_function_name():
    name = "Parabolic electron temperature profile"
    function = "parabolic_electron_temperature_profile"
    # include by function name, then exclude by function name removes it.
    included = RELATIONS.get_filtered_relations(names=[function])
    assert any(rel.name == name for rel in included)
    excluded = RELATIONS.get_filtered_relations(names=[function], exclude=[function])
    assert not any(rel.name == name for rel in excluded)
    # order by function name puts it first.
    ordered = RELATIONS.get_filtered_relations(names=[function], order=[function])
    assert ordered[0].name == name
    # order referencing an inactive (unselected) relation still errors.
    with pytest.raises(ValueError, match="inactive relation"):
        RELATIONS.get_filtered_relations(order=[function])


def test_ordered_mode_resolves_step_by_function_name():
    rho = np.linspace(0.0, 1.0, 11)
    system = RelationSystem(
        [
            Variable("rho", value=rho, fixed=True),
            Variable("T_e_avg", value=5.0, fixed=True),
            Variable("temperature_peaking", value=2.5, fixed=True),
        ],
        list(RELATIONS),
        name="ordered_by_function",
    )
    assert system.relation_by_identifier("parabolic_electron_temperature_profile").name == (
        "Parabolic electron temperature profile"
    )
    result = system.ordered(order=["parabolic_electron_temperature_profile"])
    assert not result["errors"]
    assert result["executed_relations"] == ["Parabolic electron temperature profile"]
