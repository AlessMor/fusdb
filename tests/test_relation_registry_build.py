"""Build-time validation of the relation registry.

The relation registry canonicalizes every relation against the variable
registry when it is built, rejecting alias-degenerate relations (a declared
output that resolves to one of the relation's own inputs).  These tests pin
that contract so the mistake is caught here -- at registry build -- instead of
being silently dropped when a RelationSystem is later compiled.
"""

from __future__ import annotations

import pytest

from fusdb.registry import RELATIONS, RelationRegistry
from fusdb.relation import Relation, canonicalize_relation


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
