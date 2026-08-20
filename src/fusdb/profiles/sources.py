"""Adapters for profiles supplied on external normalized coordinates.

A source profile remains immutable input data, but it is exposed to the
RelationSystem as an ordinary generated relation on fusdb's common ``rho``
grid. This keeps coordinate conversion in the existing relation/provider
graph: geometry mappings are relation inputs, so completion and Jacobian
sparsity see the dependency without a new Profile or Coordinate class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any

import numpy as np

from ..relation import Relation
from ..registry import VARIABLES
from .numerics import coordinate_average, normalized_shape, reinterpolate_profile, volume_average
from ..variable import Variable


def _source_grid(variable: Variable) -> np.ndarray:
    """Return the immutable coordinate grid attached to one supplied profile."""
    value = np.asarray(variable.input_value, dtype=float)
    if value.ndim != 1:
        raise ValueError(f"Source profile {variable.name!r} must be one-dimensional.")
    if variable.coordinate_values is not None:
        return np.asarray(variable.coordinate_values, dtype=float).copy()
    return np.linspace(0.0, 1.0, value.size)


def _initial_average(
    variable: Variable,
    records: Mapping[str, Variable],
    profile_size: int,
) -> float:
    """Best available volume-average seed for a source profile.

    The seed is not a physical constraint. When the requested coordinate and
    volume measure are already supplied, it uses them. Otherwise it falls back
    to a straight average over the source coordinate; dynamic completion then
    recomputes the shape with the actual geometry on every candidate state.
    """
    source = _source_grid(variable)
    values = np.asarray(variable.input_value, dtype=float)
    coordinate = variable.coordinate or "rho"
    rho = VARIABLES.uniform_profile_grid(profile_size)

    target: np.ndarray | None = None
    if coordinate == "rho":
        target = rho
    else:
        mapping = records.get(coordinate)
        if mapping is not None and mapping.input_value is not None:
            arr = np.asarray(mapping.input_value, dtype=float)
            if arr.ndim == 1 and arr.size == profile_size:
                target = arr

    if target is not None:
        mapped = reinterpolate_profile(values, source, target)
        weight = None
        v_norm = None
        w_record = records.get("w_V")
        if w_record is not None and w_record.input_value is not None:
            arr = np.asarray(w_record.input_value, dtype=float)
            if arr.ndim == 1 and arr.size == profile_size:
                weight = arr
        if weight is None:
            v_record = records.get("v_norm")
            if v_record is not None and v_record.input_value is not None:
                arr = np.asarray(v_record.input_value, dtype=float)
                if arr.ndim == 1 and arr.size == profile_size:
                    v_norm = arr
        try:
            return float(volume_average(mapped, rho, weight=weight, v_norm=v_norm))
        except Exception:
            pass
    return float(coordinate_average(values, source))


def _evaluate_source_profile(
    *,
    source_values: np.ndarray,
    source_coordinate: np.ndarray,
    fixed: bool,
    average: Any = None,
    mapping: Any = None,
    rho: Any,
    w_V: Any = None,
    v_norm: Any = None,
) -> np.ndarray:
    """Evaluate a generated source-profile provider.

    Keeping the evaluator at module scope removes the four near-duplicate
    closures formerly created for the coordinate/fixed combinations. Bound
    source arrays are carried by ``functools.partial`` and are picklable.
    """
    target = rho if mapping is None else mapping
    mapped = reinterpolate_profile(source_values, source_coordinate, target)
    if fixed:
        return np.asarray(mapped, dtype=float)

    weight = None if w_V is None else np.asarray(w_V, dtype=float)
    enclosed = None if weight is not None or v_norm is None else np.asarray(v_norm, dtype=float)
    _avg, shape = normalized_shape(mapped, rho, weight=weight, v_norm=enclosed)
    return np.asarray(average) * np.asarray(shape, dtype=float)


def source_profile_relation(variable: Variable, *, average_name: str | None) -> Relation:
    """Build the ordinary relation that maps one immutable source profile.

    For a movable supplied profile, ``average_name`` is the sole amplitude
    degree of freedom and the dynamically reinterpolated source curve supplies
    only the shape. For a fixed supplied profile, the absolute source values
    are mapped directly and no amplitude variable is introduced.
    """
    name = variable.name
    coordinate = variable.coordinate or "rho"
    fixed = bool(variable.fixed)
    if not fixed and average_name is None:
        raise ValueError(f"Movable source profile {name!r} has no registered volume-average variable.")

    func = partial(
        _evaluate_source_profile,
        source_values=np.asarray(variable.input_value, dtype=float).copy(),
        source_coordinate=_source_grid(variable),
        fixed=fixed,
    )
    input_names: tuple[str, ...] = ()
    argument_names: tuple[str, ...] = ()
    if not fixed:
        input_names += (str(average_name),)
        argument_names += ("average",)
    if coordinate != "rho":
        input_names += (coordinate,)
        argument_names += ("mapping",)

    return Relation(
        name=f"Source profile {name}" if coordinate == "rho" else f"Source profile {name} on {coordinate}",
        func=func,
        input_names=input_names,
        outputs=(name,),
        tags=("profile",),
        constant_names=("rho", "w_V", "v_norm"),
        dependency="generated_profile",
        function_name=f"source_profile_{name}" if coordinate == "rho" else f"source_profile_{name}_on_{coordinate}",
        argument_names=argument_names,
        source_kind="source_profile",
        source_name=name,
    )



def prepare_source_profiles(
    variables: Iterable[Variable],
    relations: Iterable[Relation],
    *,
    profile_size: int | None = None,
) -> tuple[list[Variable], tuple[Relation, ...], int]:
    """Return RelationSystem-ready variables and source-profile relations.

    External source arrays never set the common solver-grid length. The common
    size is the reactor grid when supplied, otherwise the unique size of
    ordinary profile declarations, otherwise the registry default. Each source
    profile declaration is replaced only in the RelationSystem input records by
    a missing common-grid profile; its immutable source data live in the
    generated relation. The user-facing Reactor declaration is not mutated.
    """
    records = list(variables)
    by_name = {record.name: record for record in records}
    ordinary_sizes = {
        int(record.size)
        for record in records
        if record.spec.shape == 1
        and record.input_value is not None
        and record.coordinate is None
        and record.coordinate_values is None
        and record.size is not None
    }
    if profile_size is None:
        if len(ordinary_sizes) > 1:
            raise ValueError(f"Profile sizes are incompatible: {sorted(ordinary_sizes)}.")
        common_size = next(iter(ordinary_sizes), VARIABLES.profile_size_default)
    else:
        common_size = int(profile_size)
        if common_size <= 0:
            raise ValueError("profile_size must be positive.")

    relation_list = list(relations)
    relation_outputs = {out for relation in relation_list for out in relation.output_names}
    replacements: dict[str, Variable] = {}
    additions: dict[str, Variable] = {}
    source_relations: list[Relation] = []

    for record in records:
        is_source = (
            record.spec.shape == 1
            and record.input_value is not None
            and (record.coordinate is not None or record.coordinate_values is not None)
        )
        if not is_source:
            continue
        coordinate = record.coordinate or "rho"
        if coordinate != "rho":
            mapping = by_name.get(coordinate)
            mapping_supplied = mapping is not None and mapping.input_value is not None
            if not mapping_supplied and coordinate not in relation_outputs:
                raise ValueError(
                    f"Profile {record.name!r} is defined on {coordinate!r}, but no supplied value "
                    f"or active relation provides the {coordinate!r} mapping."
                )

        average_name = VARIABLES.average_of(record.name)
        if not record.fixed and average_name is None:
            raise ValueError(
                f"Movable source profile {record.name!r} needs an average_variable in the variable registry."
            )
        replacements[record.name] = record.clone(
            value=None,
            fixed=False,
            size=common_size,
            coordinate=None,
            coordinate_values=None,
        )
        if not record.fixed and average_name is not None:
            average_record = by_name.get(average_name)
            if average_record is None:
                additions[average_name] = Variable(
                    average_name,
                    value=_initial_average(record, by_name, common_size),
                    fixed=False,
                )
            elif average_record.input_value is None and not average_record.fixed:
                replacements[average_name] = average_record.clone(
                    value=_initial_average(record, by_name, common_size)
                )
        source_relations.append(source_profile_relation(record, average_name=average_name))

    prepared = [replacements.get(record.name, record) for record in records]
    prepared.extend(additions[name] for name in sorted(additions) if name not in by_name)
    return prepared, tuple((*relation_list, *source_relations)), common_size
