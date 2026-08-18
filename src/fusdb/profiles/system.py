"""Source-aware RelationSystem construction.

This module is intentionally a function layer rather than another user-facing
class. It centralizes the preprocessing required for external profile
coordinates and geometry-dependent profile measures before handing the result
to the existing RelationSystem.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .sources import prepare_source_profiles
from ..relation import Relation
from ..registry import VARIABLES
from ..registry.coordinate_variables import PHYSICAL_COORDINATE_NAMES
from ..relationsystem import RelationSystem
from ..variable import Variable


def _drop_defaults_for_supplied_coordinates(
    variables: list[Variable], relations: tuple[Relation, ...]
) -> tuple[Relation, ...]:
    """Make a supplied physical mapping authoritative by default.

    Device coordinate relations are fallbacks. If a scenario supplies
    ``rho_tor``, ``rho_pol``, ``rho_minor``, ``rho_radial``, ``v_norm`` or
    ``w_V`` directly, retaining a tagged default producer would enforce the
    reduced fallback against the supplied equilibrium data. Drop those fallback
    providers before compilation.

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
) -> tuple[list[Variable], tuple[Relation, ...], frozenset[str]]:
    """Fold geometry-independent *tokamak* coordinate fallbacks into fixed data.

    The reduced tokamak mappings ``rho_minor=rho``, ``rho_tor=rho``,
    ``v_norm=rho**2`` and ``w_V=rho`` are deterministic migration scaffolding:
    they contain no reactor unknown and no geometry input. Keeping these static
    providers in every large tokamak reconcile changed provider ordering and
    finite-difference completion paths despite adding no information, producing
    a material runtime regression.

    This optimization is deliberately tokamak-only. The reduced stellarator and
    mirror fallbacks remain ordinary active providers. Those providers restore
    the pre-coordinate-refactor profile/average and line-average behavior on
    non-tokamak devices, and keeping them explicit makes that compatibility
    contract visible rather than silently changing those systems' relation
    graphs. A future equilibrium-derived non-tokamak mapping can supersede them
    through the normal provider mechanism.

    Geometry-dependent tokamak mappings (for example the opt-in Sauter volume
    mapping) also remain ordinary relations, so their dependencies stay visible
    to completion and Jacobian sparsity.
    """
    by_name = {variable.name: variable for variable in variables}
    rho = VARIABLES.uniform_profile_grid(profile_size)
    materialized: dict[str, Variable] = {}
    kept: list[Relation] = []

    for relation in relations:
        outputs = tuple(relation.output_names)
        static_default = (
            "default" in relation.tags
            and "tokamak" in relation.tags
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
                materialized[name] = Variable(name, value=value, fixed=True, size=profile_size)
            else:
                materialized[name] = existing.clone(value=value, fixed=True, size=profile_size)

    if not materialized:
        return variables, relations, frozenset()

    prepared = [materialized.get(variable.name, variable) for variable in variables]
    prepared.extend(materialized[name] for name in sorted(materialized) if name not in by_name)
    return prepared, tuple(kept), frozenset(materialized)


def build_relation_system(
    variables: Iterable[Variable],
    relations: Iterable[Relation],
    *,
    constraints: Any = None,
    name: str | None = None,
    profile_size: int | None = None,
) -> RelationSystem:
    """Build a RelationSystem with dynamic profile/geometry conversion enabled.

    Ordinary declarations are passed through unchanged. Profiles carrying
    ``coordinate``/``coordinate_values`` are converted to generated provider
    relations whose dependencies include the current coordinate mapping. Their
    source sample count therefore does not define the RelationSystem grid and
    their shape is recomputed whenever the mapping changes.

    Physical coordinate mappings are not solver profile degrees of freedom. A
    supplied mapping is held exactly and suppresses tagged fallback coordinate
    providers unless the declaration explicitly selects provider relations. For
    tokamaks, geometry-independent migration defaults are materialized once as
    fixed profile data so they add no artificial solver ancestry; reduced
    stellarator/mirror defaults stay explicit providers to preserve their legacy
    compatibility links. A geometry-dependent mapping remains missing until an
    active geometry relation computes it. This prevents least squares from
    inventing an arbitrary pointwise coordinate transformation while keeping
    real geometry dependencies inside the relation graph.

    Relations are never rewritten to record any of this. A materialized
    mapping is declared to the RelationSystem as a resolved constant, so the
    same equation keeps one dependency declaration in every reactor and it is
    compilation that knows the name needs no provider here.
    """
    prepared, prepared_relations, common_size = prepare_source_profiles(
        variables,
        relations,
        profile_size=profile_size,
    )
    prepared_relations = _drop_defaults_for_supplied_coordinates(prepared, prepared_relations)
    prepared, prepared_relations, static_coordinates = _materialize_static_coordinate_defaults(
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
    return RelationSystem(
        prepared,
        prepared_relations,
        constraints=constraints,
        name=name,
        resolved_constants=static_coordinates,
    )
