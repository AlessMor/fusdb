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
from .registry import VARIABLES
from .registry.coordinate_variables import PHYSICAL_COORDINATE_NAMES
from .relationsystem import RelationSystem
from .variable import Variable


def _promote_source_measure_dependencies(
    variables: list[Variable], relations: tuple[Relation, ...]
) -> tuple[Relation, ...]:
    """Expose geometry-dependent source-shape normalization in the graph.

    ``profile_sources`` keeps ``w_V``/``v_norm`` as optional keyword constants
    so the legacy source-profile API can still operate when no geometry measure
    exists. Once the scenario actually supplies or produces one of these
    measures, a movable source profile depends on it physically: remapping is
    normalized to unit volume average on every candidate state. Promote the
    preferred available measure from an optional constant to an ordinary
    relation input so provider ordering and Jacobian sparsity see that edge.
    """
    supplied = {variable.name for variable in variables if variable.input_value is not None}
    produced = {output for relation in relations for output in relation.output_names}
    available = supplied | produced
    measure = "w_V" if "w_V" in available else ("v_norm" if "v_norm" in available else None)
    if measure is None:
        return relations

    promoted: list[Relation] = []
    for relation in relations:
        average = VARIABLES.average_of(relation.source_name) if relation.source_kind == "source_profile" else None
        movable_source = average is not None and average in relation.input_names
        if not movable_source or measure not in relation.constant_names:
            promoted.append(relation)
            continue
        promoted.append(
            Relation(
                name=relation.name,
                func=relation.func,
                input_names=(*relation.input_names, measure),
                outputs=relation.outputs,
                op=relation.op,
                rhs=relation.rhs,
                tags=relation.tags,
                enforce=relation.enforce,
                constraints=relation.constraints,
                source_kind=relation.source_kind,
                source_name=relation.source_name,
                constant_names=tuple(name for name in relation.constant_names if name != measure),
                dependency=relation.dependency,
                function_name=relation.function_name,
                argument_names=(*relation.argument_names, measure),
            )
        )
    return tuple(promoted)


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
    prepared_relations = _promote_source_measure_dependencies(prepared, prepared_relations)
    return RelationSystem(
        prepared,
        prepared_relations,
        constraints=constraints,
        name=name,
    )
