"""POPCON scan container and execution logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from .reactor_class import Reactor


@dataclass
class Popcon:
    """Run and store one POPCON scan for a reactor."""

    reactor: "Reactor"
    scan_axes: dict[str, Sequence[float]]
    outputs: Iterable[str] | None = None
    constraints: Iterable[str] | None = None
    exclude_constraints: Iterable[str] | None = None
    where: dict[str, tuple[float | None, float | None]] | None = None
    chunk_size: int | None = None

    axes: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    axis_order: list[str] = field(init=False, default_factory=list)
    outputs_map: dict[str, object] = field(init=False, default_factory=dict)
    margins: dict[str, object] = field(init=False, default_factory=dict)
    allowed: np.ndarray = field(init=False)
    diagnostics: dict[str, object] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Run the scan immediately so the object is always ready to consume."""
        self.run()

    def run(self) -> "Popcon":
        """Execute scan and populate outputs/margins/allowed/diagnostics."""
        if not self.scan_axes:
            raise ValueError("scan_axes must contain at least one axis.")

        axis_order = list(self.scan_axes.keys())
        axes: dict[str, np.ndarray] = {}
        for name, values in self.scan_axes.items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1 or arr.size < 1:
                raise ValueError(f"scan_axes[{name}] must be a 1D array with at least one value.")
            var = self.reactor.variables_dict.get(name)
            if var is not None and var.ndim == 1:
                raise ValueError(f"scan axis '{name}' cannot be a profile variable.")
            axes[name] = arr

        grids = np.meshgrid(*[axes[name] for name in axis_order], indexing="ij")
        grid_shape = grids[0].shape

        base_values = {
            name: (
                None
                if var is None
                else var.current_value
                if var.current_value is not None
                else var.input_value
            )
            for name, var in self.reactor.variables_dict.items()
        }

        for name, var in self.reactor.variables_dict.items():
            if name in axis_order:
                continue
            if var.input_source is None:
                base_values[name] = None

        for name, grid in zip(axis_order, grids, strict=True):
            base_values[name] = grid

        # Build the runtime system once; it owns relation filtering and ordering.
        system = self.reactor.make_relationsystem(mode="check", verbose=False)
        active_relations = list(system.relations)

        if self.constraints is not None:
            wanted = {str(item) for item in self.constraints}
            constraint_rels = [
                relation
                for relation in active_relations
                if relation.name in wanted or any(target in wanted for target in relation.outputs)
            ]
        else:
            constraint_rels = [
                relation
                for relation in active_relations
                if "constraint" in tuple(getattr(relation, "tags", ()) or ())
            ]

        if self.exclude_constraints:
            exclude = {str(item) for item in self.exclude_constraints}
            constraint_rels = [
                relation
                for relation in constraint_rels
                if relation.name not in exclude and not set(relation.outputs).intersection(exclude)
            ]

        outputs_by_input: dict[str, set[str]] = {}
        for relation in active_relations:
            for inp in relation.input_names():
                outputs_by_input.setdefault(inp, set()).update(relation.outputs)

        dependent: set[str] = set(axis_order)
        queue = list(axis_order)
        while queue:
            name = queue.pop()
            for out in outputs_by_input.get(name, ()):
                if out in dependent:
                    continue
                dependent.add(out)
                queue.append(out)

        if self.outputs is None:
            output_names = sorted(
                {
                    *base_values.keys(),
                    *[output for relation in active_relations for output in relation.outputs],
                }
            )
        else:
            output_names = [str(name) for name in self.outputs]

        axis_set = set(axis_order)
        for name in output_names:
            if name in axis_set:
                continue
            var = self.reactor.variables_dict.get(name)
            if name in dependent or var is None or var.input_source is None:
                base_values[name] = None

        for relation in constraint_rels:
            for target in relation.outputs:
                if target not in axis_set:
                    base_values[target] = None
        if self.where:
            for name in self.where:
                if name in axis_set:
                    continue
                if name in dependent:
                    base_values[name] = None

        values_out = system.evaluate(base_values, chunk_size=self.chunk_size)

        outputs_map = {name: values_out.get(name) for name in output_names if name in values_out}
        margins_map = {
            target: values_out.get(target)
            for relation in constraint_rels
            for target in relation.outputs
            if target in values_out
        }

        allowed = np.ones(grid_shape, dtype=bool)
        for margin in margins_map.values():
            if margin is None:
                allowed &= False
                continue
            arr = np.asarray(margin)
            if arr.shape == ():
                arr = np.broadcast_to(arr, grid_shape)
            allowed &= arr <= 0

        if self.where:
            for name, (lo, hi) in self.where.items():
                arr = values_out.get(name)
                if arr is None:
                    allowed &= False
                    continue
                arr = np.asarray(arr)
                if arr.shape == ():
                    arr = np.broadcast_to(arr, grid_shape)
                if lo is not None:
                    allowed &= arr >= lo
                if hi is not None:
                    allowed &= arr <= hi

        diagnostics = {
            "fraction_allowed": float(np.mean(allowed)) if allowed.size else 0.0,
            "violation_counts": {
                name: int(np.sum(np.asarray(values) > 0))
                for name, values in margins_map.items()
                if values is not None
            },
        }

        self.axes = axes
        self.axis_order = axis_order
        self.outputs_map = outputs_map
        self.margins = margins_map
        self.allowed = allowed
        self.diagnostics = diagnostics
        return self

    def to_result(self) -> dict[str, object]:
        """Return mapping payload used by plotting and downstream consumers."""
        return {
            "axes": self.axes,
            "axis_order": self.axis_order,
            "outputs": self.outputs_map,
            "margins": self.margins,
            "allowed": self.allowed,
            "diagnostics": self.diagnostics,
        }
