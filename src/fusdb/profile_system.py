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
from .registry.coordinate_variables import PHYSICAL_COORDINATE_NAMES
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

    Physical coordinate mappings are not solver profile degrees of freedom. A
    supplied mapping is held exactly; an unsupplied mapping remains missing
    until an active geometry relation computes it. This prevents least squares
    from inventing an arbitrary pointwise coordinate transformation.
    """
    prepared, prepared_relations, _size = prepare_source_profiles(
        variables,
        relations,
        profile_size=profile_size,
    )
    prepared = [
        variable.clone(fixed=True)
        if variable.name in PHYSICAL_COORDINATE_NAMES and variable.input_value is not None and not variable.fixed
        else variable
        for variable in prepared
    ]
    return RelationSystem(
        prepared,
        prepared_relations,
        constraints=constraints,
        name=name,
    )
