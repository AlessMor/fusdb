"""Scenario variable object."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, VariableSpec, convert_value
from .utils import coerce_numeric_value, coerce_to_shape, parse_constraint_specs, unique_preserve_order, value_in_domain


@dataclass(frozen=True)
class Variable:
    """One declared scalar or profile variable: an immutable ingestion record.

    A ``Variable`` is its registry :class:`VariableSpec` (the immutable
    definition: name, aliases, unit, shape, domain, tolerances) plus one
    scenario's declaration about it. Definition metadata is read through
    ``self.spec``; scenario-local relation preference may override the registry
    ``default_relation`` without mutating the process-wide registry.

    Constructing a ``Variable`` is the ingestion event -- unit conversion,
    shape coercion and domain validation happen exactly once. A solve never
    writes back into a ``Variable``; solved state lives on the RelationSystem.

    ``default_relation`` has three states: ``None`` inherits the registry
    preference, a string/list replaces it, and an empty list explicitly disables
    the registry preference for this scenario variable. Multiple relation names
    mean simultaneous providers/constraints; provider arbitration is performed
    by the relation registry/compiler rather than here.
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
    spec: VariableSpec = field(default=None, init=False)
    input_value: Any = field(default=None, init=False)
    relations: tuple[Relation, ...] = field(default_factory=tuple, init=False)

    @property
    def aliases(self) -> tuple[str, ...]:
        """Registry aliases for this variable."""
        return self.spec.aliases

    @property
    def shape(self) -> int:
        """Registry shape: 0 for scalars, 1 for profiles."""
        return self.spec.shape

    @property
    def effective_default_relation(self) -> tuple[str, ...]:
        """Scenario relation preference, falling back to registry metadata."""
        if self.default_relation is None:
            return self.spec.default_relation
        return tuple(self.default_relation)

    def __post_init__(self) -> None:
        """Resolve registry metadata and normalize the value (the one ingestion pass)."""
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

        value = coerce_numeric_value(self.value)
        if value is not None:
            value = convert_value(value, from_unit=self.unit or spec.unit, to_unit=spec.unit)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", spec.unit)
        object.__setattr__(self, "input_value", self._copy_value(self.value))

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
            coerced, size = coerce_to_shape(self.name, self.value, is_profile=True, size=self.size)
            object.__setattr__(self, "value", coerced)
            object.__setattr__(self, "size", size)

        built: list[Relation] = []
        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):
            built.append(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_constraint_{index}",
                    enforce=enforce,
                    source_kind="variable",
                    source_name=self.name,
                )
            )
        object.__setattr__(self, "relations", tuple(built))

    def clone(self, **changes: Any) -> "Variable":
        """Return a new independently-ingested ``Variable`` with fields overridden."""
        return dataclasses.replace(self, **changes)

    def _copy_value(self, value: Any) -> Any:
        """Copy a scalar/array value."""
        if isinstance(value, np.ndarray):
            return value.copy()
        return value
