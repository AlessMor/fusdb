"""Source-aware RelationSystem construction.

This module is intentionally a function layer rather than another user-facing
class. It centralizes the only preprocessing required for external profile
coordinates before handing the result to the existing RelationSystem.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .profile_sources import prepare_source_profiles
from .relation import Relation
from .registry import VARIABLES
from .registry.coordinate_variables import PHYSICAL_COORDINATE_NAMES
from .relationsystem import RelationSystem
from .variable import Variable


def _promote_source_measure_dependencies(
    variables: list[Variable], relations: tuple[Relation, ...]
) -> tuple[Relation, ...]:
    """Expose an available geometry volume measure in movable source-profile edges.

    ``source_profile_relation`` keeps ``w_V``/``v_norm`` optional so profiles
    still work in deliberately geometry-free standalone systems. At the system
    boundary we know which mappings are supplied or produced. Promote the best
    available measure from an optional constant to a real relation input so a
    changing geometry is visible to completion and Jacobian sparsity.

    Only movable source profiles need this dependency: their immutable source
    curve is re-normalized into ``average * shape``. Fixed absolute source
    profiles are merely reinterpolated and therefore do not depend on a volume
    measure.

    ``w_V`` has precedence over ``v_norm`` because it reproduces the historical
    discrete tokamak volume average exactly and avoids differentiating an
    enclosed-volume coordinate numerically.
    """
    available = {
        variable.name for variable in variables if variable.input_value is not None
    } | {output for relation in relations for output in relation.output_names}
    measure = "w_V" if "w_V" in available else ("v_norm" if "v_norm" in available else None)
    if measure is None:
        return relations

    promoted: list[Relation] = []
    for relation in relations:
        movable_source = relation.source_kind == "source_profile" and "average" in relation.argument_names
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
                rebuild_spec=relation.rebuild_spec,
            )
        )
    return tuple(promoted)


def _drop_defaults_for_supplied_coordinates(
    variables: list[Variable], relations: tuple[Relation, ...]
) -> tuple[Relation, ...]:
    """Make a supplied physical mapping authoritative by default.

    Device coordinate relations are fallbacks. If a scenario supplies
    ``rho_tor``, ``rho_minor``, ``rho_radial``, ``v_norm`` or ``w_V`` directly,
    retaining the tagged default producer would enforce the fallback identity or
    self-similar mapping against the supplied equilibrium data. Drop those
    fallback producers before compilation.

    A non-empty variable-local ``default_relation`` is the explicit opt-in to
    keep provider relations alongside supplied data, so users can deliberately
    certify/reconcile an imported mapping against a chosen model.
    """
    authoritative = {
        variable.name
        for variable in variables
        if variable.name in PHYSICAL_COORDINATE_NAMES
        and variable.input_value is not None
        and not variable.default_relation
    }
    if not authoritative:
        return relations
    return tuple(
        relation
        for relation in relations
        if not (
            "default" in relation.tags
            and bool(authoritative.intersection(relation.output_names))
        )
    )


def _materialize_static_coordinate_defaults(
    variables: list[Variable],
    relations: tuple[Relation, ...],
    profile_size: int,
) -> tuple[list[Variable], tuple[Relation, ...]]:
    """Fold geometry-independent coordinate fallbacks into fixed profile data.

    Identity/self-similar device defaults such as ``rho_minor=rho`` and
    ``w_V=rho`` are deterministic framework data: they contain no reactor
    unknown and no geometry input. Keeping them as completion providers changes
    provider ordering and finite-difference completion paths even though their
    values are constant. On large reconcile problems that altered the numerical
    trajectory enough to cause a multi-fold runtime regression.

    Only a very narrow class is folded here: tagged ``default`` relations whose
    outputs are all physical coordinate variables, with no ordinary inputs and
    no constants except the framework ``rho`` grid. Geometry-dependent mappings
    (for example the opt-in Sauter volume mapping) remain ordinary relations, so
    their dependencies stay visible to completion and Jacobian sparsity.

    Explicitly supplied mappings and declarations with a local
    ``default_relation`` are also left alone; those are user-selected provenance
    choices rather than implicit fallbacks.
    """
    by_name = {variable.name: variable for variable in variables}
    rho = VARIABLES.uniform_profile_grid(profile_size)
    materialized: dict[str, Variable] = {}
    kept: list[Relation] = []

    for relation in relations:
        outputs = tuple(relation.output_names)
        static_default = (
            "default" in relation.tags
            and bool(outputs)
            and set(outputs) <= PHYSICAL_COORDINATE_NAMES
            and not relation.input_names
            and set(relation.constant_names) <= {"rho"}
        )
        if not static_default:
            kept.append(relation)
            continue

        declared = [by_name.get(name) for name in outputs]
        if any(
            variable is not None
            and (variable.input_value is not None or variable.default_relation is not None)
            for variable in declared
        ):
            kept.append(relation)
            continue

        mapped = relation.output_map(relation.evaluate({"rho": rho}))
        for name in outputs:
            value = np.asarray(mapped[name], dtype=float)
            existing = by_name.get(name)
            if existing is None:
                materialized[name] = Variable(
                    name,
                    value=value,
                    fixed=True,
                    size=profile_size,
                )
            else:
                materialized[name] = existing.clone(
                    value=value,
                    fixed=True,
                    size=profile_size,
                )

    if not materialized:
        return variables, relations

    prepared = [materialized.get(variable.name, variable) for variable in variables]
    prepared.extend(
        materialized[name]
        for name in sorted(materialized)
        if name not in by_name
    )
    return prepared, tuple(kept)


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
    supplied mapping is held exactly and suppresses tagged fallback coordinate
    providers unless the declaration explicitly selects provider relations. An
    unsupplied geometry-independent default is materialized once as fixed
    profile data; a geometry-dependent mapping remains missing until an active
    geometry relation computes it. This prevents least squares from inventing an
    arbitrary pointwise coordinate transformation while keeping real geometry
    dependencies inside the relation graph.
    """
    prepared, prepared_relations, common_size = prepare_source_profiles(
        variables,
        relations,
        profile_size=profile_size,
    )
    prepared_relations = _drop_defaults_for_supplied_coordinates(prepared, prepared_relations)
    prepared, prepared_relations = _materialize_static_coordinate_defaults(
        prepared,
        prepared_relations,
        common_size,
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
