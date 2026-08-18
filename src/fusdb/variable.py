"""Scenario variable object."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .relation import Relation, build_constraint_relations
from .registry import VARIABLES, VariableSpec, convert_value
from .numerics import coerce_numeric_value, coerce_to_shape, unique_preserve_order, value_in_domain


@dataclass(frozen=True)
class Variable:
    """One declared scalar or profile variable: an immutable ingestion record.

    A ``Variable`` is its registry :class:`VariableSpec` plus one scenario's
    declaration about it. Constructing a ``Variable`` is the ingestion event:
    unit conversion, shape coercion and domain validation happen exactly once.
    A solve never writes back into a ``Variable``; solved state lives on the
    RelationSystem.

    ``default_relation`` has three states: ``None`` inherits the registry
    preference, a string/list replaces it, and an empty list explicitly disables
    the registry preference for this scenario variable.

    Profile declarations may additionally retain their immutable source
    coordinate. ``coordinate`` names the physical normalized coordinate on
    which the supplied samples were tabulated (``rho`` means the common fusdb
    grid). ``coordinate_values`` stores that source grid and may have a different
    length from the RelationSystem grid. These are ingestion data, not solver
    unknowns. The RelationSystem is responsible for reinterpolating them through
    the current geometry mapping during completion.
    """

    name: str
    value: Any = None
    unit: str | None = None
    rel_tol: float | None = None
    abs_tol: float | None = None
    fixed: bool = False
    size: int | None = None
    constraints: Any = None
    default_relation: tuple[str, ...] | str | list[str] | None = None
    coordinate: str | None = None
    coordinate_values: Any = None
    spec: VariableSpec = field(default=None, init=False)
    input_value: Any = field(default=None, init=False)
    relations: tuple[Relation, ...] = field(default_factory=tuple, init=False)

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.spec.aliases

    @property
    def shape(self) -> int:
        return self.spec.shape

    @property
    def effective_default_relation(self) -> tuple[str, ...]:
        if self.default_relation is None:
            return self.spec.default_relation
        return tuple(self.default_relation)

    @property
    def has_source_grid(self) -> bool:
        """Whether this declaration carries explicit source-coordinate samples."""
        return self.coordinate_values is not None

    def __post_init__(self) -> None:
        spec = VARIABLES.get(self.name)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "name", spec.name)
        object.__setattr__(self, "rel_tol", spec.rel_tol if self.rel_tol is None else float(self.rel_tol))
        object.__setattr__(self, "abs_tol", spec.abs_tol if self.abs_tol is None else float(self.abs_tol))

        local_default = self.default_relation
        if local_default is not None:
            if isinstance(local_default, str):
                local_default = (local_default,)
            else:
                local_default = unique_preserve_order(local_default)
            object.__setattr__(self, "default_relation", tuple(str(name) for name in local_default))

        coordinate = self.coordinate
        coordinate_values = self.coordinate_values
        if self.shape == 0 and (coordinate is not None or coordinate_values is not None):
            raise ValueError(f"Scalar variable {spec.name!r} cannot define a profile coordinate.")
        if coordinate_values is not None and coordinate is None:
            coordinate = "rho"
        if coordinate is not None:
            coordinate = str(coordinate).strip()
            if not coordinate:
                raise ValueError(f"Variable {spec.name!r} coordinate cannot be empty.")
            if coordinate != "rho":
                try:
                    coordinate = VARIABLES.resolve(coordinate)
                except KeyError as exc:
                    raise ValueError(
                        f"Variable {spec.name!r} uses unknown source coordinate {coordinate!r}."
                    ) from exc
            object.__setattr__(self, "coordinate", coordinate)

        value = coerce_numeric_value(self.value)
        if value is not None:
            value = convert_value(value, from_unit=self.unit or spec.unit, to_unit=spec.unit)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", spec.unit)

        if self.size is not None:
            size = int(self.size)
            if size <= 0:
                raise ValueError(f"Variable {self.name!r} size must be positive.")
            object.__setattr__(self, "size", size)
        if self.shape == 0 and self.size is not None:
            raise ValueError(f"Scalar variable {self.name!r} cannot define a profile size.")
        if self.value is not None and not value_in_domain(self.value, spec.domain):
            raise ValueError(f"Variable {self.name!r} value is outside domain {spec.domain!r}.")
        if self.shape == 1 and self.value is not None:
            # An explicit source grid owns the supplied profile length. It must
            # not be coerced to the reactor's common solver-grid size here.
            source_size = None if coordinate_values is not None else self.size
            coerced, inferred_size = coerce_to_shape(self.name, self.value, is_profile=True, size=source_size)
            object.__setattr__(self, "value", coerced)
            object.__setattr__(self, "size", inferred_size)

        if coordinate_values is not None:
            source = np.asarray(coordinate_values, dtype=float)
            if source.ndim != 1 or source.size < 2:
                raise ValueError(
                    f"Variable {self.name!r} coordinate_values must be a one-dimensional grid with at least two points."
                )
            if not np.all(np.isfinite(source)) or np.any(np.diff(source) <= 0.0):
                raise ValueError(
                    f"Variable {self.name!r} coordinate_values must be finite and strictly increasing."
                )
            if self.value is not None:
                arr = np.asarray(self.value)
                if arr.ndim != 1 or arr.shape[0] != source.shape[0]:
                    raise ValueError(
                        f"Variable {self.name!r} profile length {arr.shape[0] if arr.ndim else 1} "
                        f"does not match coordinate_values length {source.shape[0]}."
                    )
            object.__setattr__(self, "coordinate_values", source.copy())
            object.__setattr__(self, "size", int(source.size))

        object.__setattr__(self, "input_value", self.value.copy() if isinstance(self.value, np.ndarray) else self.value)

        object.__setattr__(
            self,
            "relations",
            build_constraint_relations(
                self.constraints,
                name_prefix=f"{self.name}_constraint",
                source_kind="variable",
                source_name=self.name,
            ),
        )

    def clone(self, **changes: Any) -> "Variable":
        return dataclasses.replace(self, **changes)

