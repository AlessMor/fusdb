"""Scenario variable object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, convert_value
from .utils import coerce_numeric_value, parse_constraint_specs, value_in_domain


@dataclass
class Variable:
    """One active scalar or profile variable.

    Args:
        name: Canonical name or alias.
        value: Scalar, 1D profile, Pint quantity, or None.
        unit: Unit of ``value``. If omitted, registry default is assumed.
        rel_tol: Relative tolerance override.
        fixed: Whether solve modes may change this value.
        size: Profile length for 1D variables.
        constraints: Additional local constraints.
    """

    name: str
    value: Any = None
    unit: str | None = None
    rel_tol: float | None = None
    fixed: bool = False
    size: int | None = None
    constraints: Any = None
    shape: int = field(default=0, init=False)
    source: str = field(default="missing", init=False)
    freedom: str = field(default="free", init=False)
    validity: str = field(default="unresolved", init=False)
    reference_value: Any = field(default=None, init=False)
    relations: tuple[Relation, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        """Resolve registry metadata and convert the input value to canonical units."""
        spec = VARIABLES.get(self.name)
        self.name = spec.name
        self.shape = spec.shape
        self.rel_tol = spec.rel_tol if self.rel_tol is None else float(self.rel_tol)
        input_unit = self.unit or spec.unit
        self.value = coerce_numeric_value(self.value)
        if self.value is not None:
            self.value = convert_value(self.value, from_unit=input_unit, to_unit=spec.unit)
        self.unit = spec.unit
        self.reference_value = self._copy_value(self.value)
        self.source = "given" if self.value is not None else "missing"
        self.freedom = "fixed" if self.fixed else "free"

        if self.size is not None:
            self.size = int(self.size)
            if self.size <= 0:
                raise ValueError(f"Variable {self.name!r} size must be positive.")
        if self.shape == 0 and self.size is not None:
            raise ValueError(f"Scalar variable {self.name!r} cannot define a profile size.")
        if self.value is not None and not value_in_domain(self.value, spec.domain):
            raise ValueError(f"Variable {self.name!r} value is outside domain {spec.domain!r}.")

        if self.shape == 1 and self.value is not None:
            arr = np.asarray(self.value)
            if arr.ndim == 0:
                if self.size is None:
                    self.value = float(arr)
                else:
                    self.value = np.full(self.size, float(arr))
            elif arr.ndim == 1:
                if self.size is None:
                    self.size = int(arr.shape[0])
                elif self.size != int(arr.shape[0]):
                    raise ValueError(f"Variable {self.name!r} size mismatch: {self.size} vs {arr.shape[0]}.")
                self.value = arr.astype(float)
            else:
                raise ValueError(f"Profile variable {self.name!r} value must be scalar or 1D.")

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
        """Return a fresh variable with selected field overrides."""
        data = {
            "name": self.name,
            "value": self._copy_value(self.value),
            "unit": self.unit,
            "rel_tol": self.rel_tol,
            "fixed": self.fixed,
            "size": self.size,
            "constraints": self.constraints,
        }
        data.update(changes)
        return Variable(**data)

    def set_value(self, value: Any, *, source: str | None = None) -> None:
        """Set a canonical-unit value and update source metadata.

        Args:
            value: Scalar or one-dimensional profile value in canonical units.
            source: Optional source label.
        """
        if self.shape == 1 and value is not None:
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0 and self.size is not None:
                self.value = np.full(self.size, float(arr))
            elif arr.ndim == 1:
                if self.size is None:
                    self.size = int(arr.shape[0])
                elif self.size != int(arr.shape[0]):
                    raise ValueError(f"Variable {self.name!r} size mismatch: {self.size} vs {arr.shape[0]}.")
                self.value = arr.copy()
            else:
                raise ValueError(f"Profile variable {self.name!r} value must be scalar or 1D.")
        else:
            self.value = self._copy_value(value)
        if source is not None:
            self.source = source

    def as_dict(self) -> dict[str, Any]:
        """Return a simple serializable view of the variable."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "shape": self.shape,
            "fixed": self.fixed,
            "rel_tol": self.rel_tol,
            "source": self.source,
            "freedom": self.freedom,
            "validity": self.validity,
        }

    @staticmethod
    def _copy_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        return value
