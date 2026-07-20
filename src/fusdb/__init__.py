"""FusDB public API."""

from __future__ import annotations

from .io import load_result, save_result
from .relation import Relation, RelationSolveError, RelationUnderdeterminedError, RelationVerificationError, constraint_from_expression, relation
from .relationsystem import RelationSystem
from .variable import Variable
from .reactor import Reactor, solve_reactors
from .plotting.tables import SolvedColumn, render_table, variable_table_data
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


def __dir__() -> list[str]:
    """Include dynamically exported relation functions in API discovery."""
    relations = (rel.function_name for rel in RELATIONS)
    return sorted({*globals(), *relations})


__all__ = [
    "Relation",
    "RelationSolveError",
    "RelationUnderdeterminedError",
    "RelationVerificationError",
    "RelationSystem",
    "Reactor",
    "Variable",
    "SolvedColumn",
    "load_result",
    "save_result",
    "solve_reactors",
    "variable_table_data",
    "render_table",
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
