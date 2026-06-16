"""Scenario variable object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, convert_value
from .utils import coerce_numeric_value, coerce_to_shape, parse_constraint_specs, value_in_domain


@dataclass
class Variable:
    """One active scalar or profile variable.

    Args:
        name: Canonical variable name or alias.
        value: Scalar, one-dimensional profile, or None.
        unit: Unit of ``value``. If omitted, the registry default is assumed.
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
    aliases: tuple[str, ...] = field(default_factory=tuple, init=False)
    shape: int = field(default=0, init=False)
    input_value: Any = field(default=None, init=False)
    relations: tuple[Relation, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        """Resolve registry metadata and normalize the value."""
        spec = VARIABLES.get(self.name)
        self.name = spec.name
        self.aliases = tuple(spec.aliases)
        self.shape = spec.shape
        self.rel_tol = spec.rel_tol if self.rel_tol is None else float(self.rel_tol)
        self.abs_tol = spec.abs_tol if self.abs_tol is None else float(self.abs_tol)
        input_unit = self.unit or spec.unit
        self.value = coerce_numeric_value(self.value)
        if self.value is not None:
            self.value = convert_value(self.value, from_unit=input_unit, to_unit=spec.unit)
        self.unit = spec.unit
        self.input_value = self._copy_value(self.value)

        # Validate profile shape and physical domain.
        if self.size is not None:
            self.size = int(self.size)
            if self.size <= 0:
                raise ValueError(f"Variable {self.name!r} size must be positive.")
        if self.shape == 0 and self.size is not None:
            raise ValueError(f"Scalar variable {self.name!r} cannot define a profile size.")
        if self.value is not None and not value_in_domain(self.value, spec.domain):
            raise ValueError(f"Variable {self.name!r} value is outside domain {spec.domain!r}.")
        if self.shape == 1 and self.value is not None:
            arr = np.asarray(self.value, dtype=float)
            if arr.ndim == 0:
                self.value = float(arr) if self.size is None else np.full(self.size, float(arr))
            elif arr.ndim == 1:
                if self.size is None:
                    self.size = int(arr.shape[0])
                elif self.size != int(arr.shape[0]):
                    raise ValueError(f"Variable {self.name!r} size mismatch: {self.size} vs {arr.shape[0]}.")
                self.value = arr.astype(float)
            else:
                raise ValueError(f"Profile variable {self.name!r} value must be scalar or 1D.")

        # Variable constraints are relation guards attached to the variable.
        built: list[Relation] = []
        for index, (text, enforce) in enumerate(parse_constraint_specs(spec.constraints)):
            built.append(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_registry_constraint_{index}",
                    enforce=enforce,
                    source_kind="variable",
                    source_name=self.name,
                )
            )
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
        self.relations = tuple(built)

    def clone(self, **changes: Any) -> "Variable":
        """Return a fresh variable with selected field overrides.

        Args:
            **changes: Constructor field overrides.

        Returns:
            New Variable instance.
        """
        data = {
            "name": self.name,
            "value": self._copy_value(self.input_value),
            "unit": self.unit,
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "fixed": self.fixed,
            "size": self.size,
            "constraints": self.constraints,
        }
        data.update(changes)
        return Variable(**data)

    def _normalize_value(self, value: Any) -> Any:
        """Normalize a canonical-unit value to this variable shape."""
        if value is None:
            return None
        coerced, self.size = coerce_to_shape(
            self.name, value, is_profile=self.shape == 1, size=self.size
        )
        return coerced

    def set_input(self, value: Any) -> None:
        """Set the user/input value in canonical units.

        Args:
            value: New canonical-unit value.
        """
        normalized = self._normalize_value(value)
        spec = VARIABLES.get(self.name)
        if normalized is not None and not value_in_domain(normalized, spec.domain, zero_tol=0.0):
            raise ValueError(f"Variable {self.name!r} value is outside domain {spec.domain!r}.")
        self.input_value = self._copy_value(normalized)
        self.value = self._copy_value(normalized)

    def set_value(self, value: Any) -> None:
        """Set the current public value in canonical units.

        Args:
            value: New canonical-unit value.
        """
        normalized = self._normalize_value(value)
        spec = VARIABLES.get(self.name)
        if normalized is not None and not value_in_domain(normalized, spec.domain, zero_tol=0.0):
            raise ValueError(f"Variable {self.name!r} value is outside domain {spec.domain!r}.")
        self.value = self._copy_value(normalized)

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
