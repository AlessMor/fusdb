"""FusDB public API."""

from __future__ import annotations

from .relation import Relation, RelationSolveError, RelationUnderdeterminedError, RelationVerificationError, constraint_from_expression, relation
from .relationsystem import RelationSystem
from .variable import Variable
from .reactor import Reactor
from .registry import RELATIONS, SPECIES, TAGS, VARIABLES, RelationRegistry, SpeciesRegistry, TagRegistry, VariableRegistry


def __getattr__(name: str) -> Relation:
    """Expose decorated relations as ``fusdb.<function_name>``.

    Args:
        name: Decorated function name or relation name.

    Returns:
        Relation object.
    """
    try:
        return RELATIONS.get(name)
    except Exception as exc:
        raise AttributeError(name) from exc


__all__ = [
    "Relation",
    "RelationSolveError",
    "RelationUnderdeterminedError",
    "RelationVerificationError",
    "RelationSystem",
    "Reactor",
    "Variable",
    "constraint_from_expression",
    "relation",
    "RELATIONS",
    "SPECIES",
    "TAGS",
    "VARIABLES",
    "RelationRegistry",
    "SpeciesRegistry",
    "TagRegistry",
    "VariableRegistry",
]
