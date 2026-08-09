"""Source-aware RelationSystem construction.

This module is intentionally a function layer rather than another user-facing
class. It centralizes the only preprocessing required for external profile
coordinates before handing the result to the existing RelationSystem.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .profile_sources import prepare_source_profiles
from .relation import Relation
from .relationsystem import RelationSystem
from .variable import Variable


def build_relation_system(
    variables: Iterable[Variable],
    relations: Iterable[Relation],
    *,
    constraints: Any = None,
    name: str | None = None,
    profile_size: int | None = None,
) -> RelationSystem:
    """Build a RelationSystem with dynamic source-profile conversion enabled.

    Ordinary declarations are passed through unchanged. Profiles carrying
    ``coordinate``/``coordinate_values`` are converted to generated provider
    relations whose dependencies include the current coordinate mapping. Their
    source sample count therefore does not define the RelationSystem grid and
    their shape is recomputed whenever the mapping changes.
    """
    prepared, prepared_relations, _size = prepare_source_profiles(
        variables,
        relations,
        profile_size=profile_size,
    )
    return RelationSystem(
        prepared,
        prepared_relations,
        constraints=constraints,
        name=name,
    )
