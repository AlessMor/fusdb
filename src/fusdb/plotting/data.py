"""Backend-neutral data models used by fusdb plotters and table renderers.

These objects deliberately depend only on NumPy.  Scientific modules prepare
them once, then the matplotlib, Bokeh, HTML, or text presentation layer can
consume the same data without re-evaluating relations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    return array


@dataclass(frozen=True)
class Curve:
    """One labelled x-y series with presentation hints and scientific metadata."""

    x: ArrayLike
    y: ArrayLike
    label: str
    style: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    columns: Mapping[str, ArrayLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = _array(self.x, name="Curve.x")
        y = _array(self.y, name="Curve.y")
        if x.size != y.size:
            raise ValueError(f"Curve {self.label!r} has x/y lengths {x.size} and {y.size}.")
        columns = {name: _array(values, name=f"Curve.columns[{name!r}]") for name, values in self.columns.items()}
        if any(values.size != x.size for values in columns.values()):
            raise ValueError(f"Curve {self.label!r} columns must have the same length as x.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "style", dict(self.style))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "columns", columns)

    def source_data(self) -> dict[str, NDArray[np.float64]]:
        """Return column-oriented data suitable for a Bokeh ``ColumnDataSource``."""
        return {"x": self.x, "y": self.y, **self.columns}


@dataclass(frozen=True)
class CurveSet:
    """A group of curves sharing axes and plot-level metadata."""

    curves: Sequence[Curve]
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    xscale: str = "linear"
    yscale: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "curves", tuple(self.curves))
        if self.xscale not in {"linear", "log"} or self.yscale not in {"linear", "log"}:
            raise ValueError("CurveSet scales must be 'linear' or 'log'.")


@dataclass(frozen=True)
class FieldMap:
    """A 2-D coordinate mesh and one or more aligned scalar fields."""

    x: ArrayLike
    y: ArrayLike
    fields: Mapping[str, ArrayLike]
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError("FieldMap x and y must be same-shaped two-dimensional meshes.")
        fields = {name: np.asarray(values, dtype=float) for name, values in self.fields.items()}
        invalid = [name for name, values in fields.items() if values.shape != x.shape]
        if invalid:
            raise ValueError(f"FieldMap fields do not match mesh shape {x.shape}: {', '.join(invalid)}.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "metadata", dict(self.metadata))
