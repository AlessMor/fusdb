"""Scenario variable object."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, VariableSpec, convert_value
from .utils import coerce_numeric_value, coerce_to_shape, parse_constraint_specs, value_in_domain


@dataclass(frozen=True)
class Variable:
    """One declared scalar or profile variable: an immutable ingestion record.

    A ``Variable`` is its registry :class:`VariableSpec` (the immutable
    definition: name, aliases, unit, shape, domain, tolerances) plus one
    scenario's *declaration* about it (canonical value, ``fixed``, tolerance
    overrides, profile size, local guards).  Definition metadata is read
    through ``self.spec``; it is never copied onto the instance, so a
    variable cannot drift out of sync with its registry.

    Constructing a ``Variable`` *is* the ingestion event -- unit conversion,
    shape coercion, domain validation -- and the instance is frozen
    immediately afterward, so possessing one is proof that event happened
    exactly once.  There are no setters: a changed declaration is a new
    ``Variable`` (see :meth:`clone`), never a mutation of this one.  A solve
    never writes back into a ``Variable``; solved state lives on the
    ``RelationSystem`` that ran it (``reactor.last_system``), and
    :class:`fusdb.reactor.SolvedVariable` is the read-through view over both.

    Args:
        name: Canonical variable name or alias.
        value: Scalar, one-dimensional profile, or None.
        unit: Unit of the supplied ``value``. If omitted, the registry default is assumed.
        rel_tol: Relative tolerance override.
        fixed: Whether solve modes may change this value.
        size: Profile length for one-dimensional variables.
        constraints: Additional local constraints or applicability guards.
    """

    name: str
    value: Any = None
    unit: str | None = None
    rel_tol: float | None = None
    abs_tol: float | None = None
    fixed: bool = False
    size: int | None = None
    constraints: Any = None
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

    def __post_init__(self) -> None:
        """Resolve registry metadata and normalize the value (the one ingestion pass)."""
        spec = VARIABLES.get(self.name)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "name", spec.name)
        object.__setattr__(self, "rel_tol", spec.rel_tol if self.rel_tol is None else float(self.rel_tol))
        object.__setattr__(self, "abs_tol", spec.abs_tol if self.abs_tol is None else float(self.abs_tol))
        value = coerce_numeric_value(self.value)
        if value is not None:
            value = convert_value(value, from_unit=self.unit or spec.unit, to_unit=spec.unit)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", spec.unit)  # value is now in canonical units
        object.__setattr__(self, "input_value", self._copy_value(self.value))

        # Validate profile shape and physical domain.
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
            # ``input_value`` was already captured above, before this
            # coercion -- matching the pre-freeze ordering exactly (a
            # scalar supplied for a profile variable keeps a scalar
            # ``input_value`` while ``value`` is the broadcast array).
            coerced, size = coerce_to_shape(self.name, self.value, is_profile=True, size=self.size)
            object.__setattr__(self, "value", coerced)
            object.__setattr__(self, "size", size)

        # Record-local constraints are relation guards attached to this input;
        # registry-level constraint guards live on the spec.
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
        """Return a new, independently-ingested ``Variable`` with fields overridden.

        This is the only way to change a declaration: ``var.clone(value=3.3)``
        or ``var.clone(fixed=True)``.  Every field not named in ``changes``
        carries over from ``self``; the result goes through
        :meth:`__post_init__` fresh (full unit conversion and validation),
        exactly as if newly constructed.
        """
        return dataclasses.replace(self, **changes)

    def _copy_value(self, value: Any) -> Any:
        """Copy a scalar/array value.

        Args:
            value: Value to copy.

        Returns:
            Independent copy where appropriate.
        """
        if isinstance(value, np.ndarray):
            return value.copy()
        return value
