"""Reactor class for loading and solving fusion reactor design specifications.

This module provides the Reactor class which loads reactor specifications from
YAML files, manages variables, filters applicable relations, and orchestrates
the solving process.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import logging
import warnings
from .relation_class import Relation
from .relationsystem_class import RelationSystem
from .relation_util import get_filtered_relations, relation_input_names
from .utils import normalize_tag, normalize_tags_to_tuple
from .variable_class import Variable

logger = logging.getLogger(__name__)

@dataclass
class Reactor:
    """Container for fusion reactor specifications, variables, and solving logic.
    
    A Reactor represents a fusion reactor design with its parameters (geometry, plasma properties, etc.) and the physics relations that govern it. 
    The class can load a configuration (currently from YAML), filter applicable relations based on tags, and solve for unknown variables.
    
    Attributes:
        path: Path to the source reactor file.
        id: Unique reactor identifier (e.g., "ARC_2015", "DEMO_2022").
        name: Reactor name.
        organization: Organization (industrial or academic) developing the reactor.
        country: Country where reactor is being developed/built.
        year: Design year.
        doi: DOI reference to published design.
        notes: Additional notes.
        tags: Classification tags (e.g., "tokamak", "hmode",...). Used for relations filtering.
        solving_order: Ordered domains/relations to solve (e.g., ["geometry", "fusion_power"]).
        solver_mode: How to handle computed vs input values:
            - "overwrite": Replace input values with computed (default)
            - "check": Only check consistency, don't modify
        verbose: Enable detailed solver logging.
        relations: Filtered list of applicable relations for this reactor.
        variables_dict: Dictionary of Variable objects keyed by name.
    """
    path: Path | None = None
    id: str | None = None
    name: str | None = None
    organization: str | None = None
    country: str | None = None
    year: int | None = None
    doi: str | None = None
    notes: str | None = None
    tags: list[str] = field(default_factory=list)
    solving_order: list[str] = field(default_factory=list)
    solver_mode: str = "overwrite"
    verbose: bool = False
    relations: list[Relation] = field(default_factory=list)
    default_relations: list[Relation] = field(default_factory=list)
    variables_dict: dict[str, Variable] = field(default_factory=dict)
    _log: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize per-instance logger with context."""
        self._log = logger.getChild(self.__class__.__name__)
        self._log.setLevel(logging.INFO if self.verbose else logging.WARNING)

    @classmethod # can call Reactor.from_yaml() directly, without an instance
    def from_yaml(cls, path_like: str | Path) -> "Reactor":
        """Create a Reactor object from a YAML file.
        
        Parses the YAML file containing reactor metadata, tags, variables, and
        solver settings. Automatically filters relations based on tags and
        creates a fully initialized Reactor object ready for solving.
        
        Args:
            path_like: Path to reactor.yaml file or directory containing it.
        
        Returns:
            Reactor object with variables and relations loaded.
            
        Example:
            >>> reactor = Reactor.from_yaml('reactors/ARC_2015/reactor.yaml')
            >>> reactor.solve()
        """
        from .reactor_util import reactor_from_yaml

        return reactor_from_yaml(path_like, cls=cls)

    def _ordered_relations(self) -> Iterable[Relation]:
        """Yield applicable relations in the order the solver should consume them.

        Yields:
            Relation objects in solve order without duplicates.
        """
        # Start from the relations that match this reactor state.
        variable_names = self.variables_dict.keys()
        variable_methods = [var.method for var in self.variables_dict.values() if var.method]
        base_relations = list(
            get_filtered_relations(
                self.tags,
                variable_names,
                variable_methods,
                extra_relations=self.default_relations,
            )
        )
        rel_by_name = {rel.name: rel for rel in base_relations}

        # Without an explicit domain order, prefer the registry solving order.
        if not self.solving_order:
            from .registry import load_allowed_tags

            domain_order = {
                normalize_tag(name): idx
                for idx, name in enumerate((load_allowed_tags().get("solving_order", {}) or {}).keys())
            }
            if domain_order:
                indexed = list(enumerate(base_relations))
                indexed.sort(
                    key=lambda item: (
                        min(
                            (
                                domain_order[normalize_tag(tag)]
                                for tag in (getattr(item[1], "tags", ()) or ())
                                if normalize_tag(tag) in domain_order
                            ),
                            default=len(domain_order),
                        ),
                        item[0],
                    )
                )
                yield from [rel for _, rel in indexed]
                return

            yield from base_relations
            return

        # With an explicit domain order, walk the order top-to-bottom and
        # deduplicate relation objects as they are emitted.
        seen: set[Relation] = set()
        for item in self.solving_order:
            if item in rel_by_name:
                rel = rel_by_name[item]
                if rel not in seen:
                    seen.add(rel)
                    yield rel
                continue
            domain_tags = (*self.tags, normalize_tag(item))
            for rel in get_filtered_relations(
                domain_tags,
                variable_names,
                variable_methods,
                extra_relations=self.default_relations,
            ):
                if rel in seen:
                    continue
                seen.add(rel)
                yield rel

    def solve(
        self,
        mode: str | None = None,
        *,
        verbose: bool | None = None,
        enforce_constraint_tags: Iterable[str] | None = None,
        enforce_constraint_names: Iterable[str] | None = None,
    ) -> None:
        """Solve for unknown reactor variables using physics relations.
        
        Executes the iterative solver to compute unknown variables from known ones.
        If solving_order is specified, solves each domain in sequence.
        
        Args:
            mode: Override solver mode ("overwrite" or "check").
                 If None, uses self.solver_mode.
            verbose: Override verbosity setting. If None, uses self.verbose.
            enforce_constraint_tags: Relation tags whose soft constraints should be enforced.
            enforce_constraint_names: Relation names/outputs or variable names whose soft constraints should be enforced.
        Example:
            >>> reactor = Reactor.from_yaml('reactors/ARC_2015/reactor.yaml')
            >>> reactor.solve(verbose=True)  # Show detailed solving progress
        """
        # Resolve runtime solver settings first so the rest of the method reads
        # from one normalized set of values.
        solver_mode = mode or self.solver_mode
        verbosity = self.verbose if verbose is None else verbose
        self._log.setLevel(logging.INFO if verbosity else logging.WARNING)
        self._log.log(
            logging.INFO,
            "Solving reactor %s (%s) with mode=%s",
            self.id or "unknown",
            self.name or "unknown",
            solver_mode,
            stacklevel=2,
        )
        constraint_tags_tuple = tuple(enforce_constraint_tags or ())
        constraint_names_tuple = tuple(enforce_constraint_names or ())

        # Solve everything in one shot when no domain ordering is requested.
        if not self.solving_order:
            rels = list(self.relations)
            self._log.log(
                logging.INFO,
                "Solving %d relations (no domains)",
                len(rels),
                stacklevel=2,
            )
            system = RelationSystem(
                rels,
                list(self.variables_dict.values()),
                mode=solver_mode,
                verbose=verbosity,
                enforce_constraint_tags=constraint_tags_tuple,
                enforce_constraint_names=constraint_names_tuple,
            )
            system.solve()
            self.variables_dict.update(system.variables_dict)
            return

        # Reuse the current reactor state to resolve each requested domain in order.
        variable_names = self.variables_dict.keys()
        variable_methods = [var.method for var in self.variables_dict.values() if var.method]
        rel_by_name = {rel.name: rel for rel in self._ordered_relations()}

        # Warn early when two requested domains are both able to drive the same output.
        if len(self.solving_order) > 1:
            seen_outputs: dict[str, set[str]] = {}
            for item in self.solving_order:
                if item in rel_by_name:
                    outputs = set(rel_by_name[item].outputs)
                else:
                    domain_tags = (*self.tags, normalize_tag(item))
                    outputs = {
                        output
                        for rel in get_filtered_relations(
                            domain_tags,
                            variable_names,
                            variable_methods,
                            extra_relations=self.default_relations,
                        )
                        for output in rel.outputs
                    }
                for other_name, other_outputs in seen_outputs.items():
                    overlap = outputs & other_outputs
                    if overlap:
                        warnings.warn(
                            f"Relation domains '{other_name}' and '{item}' both solve {sorted(overlap)}",
                            UserWarning,
                        )
                seen_outputs[item] = outputs

        # Solve each requested domain against the latest variable values.
        for item in self.solving_order:
            if item in rel_by_name:
                rels = [rel_by_name[item]]
            else:
                rels = list(
                    get_filtered_relations(
                        (*self.tags, normalize_tag(item)),
                        variable_names,
                        variable_methods,
                        extra_relations=self.default_relations,
                    )
                )
            self._log.log(
                logging.INFO,
                "Solving domain %s with %d relations",
                item,
                len(rels),
                stacklevel=2,
            )
            system = RelationSystem(
                rels,
                list(self.variables_dict.values()),
                mode=solver_mode,
                verbose=verbosity,
                enforce_constraint_tags=constraint_tags_tuple,
                enforce_constraint_names=constraint_names_tuple,
            )
            system.solve()
            self.variables_dict.update(system.variables_dict)

    def diagnose(self) -> dict[str, object]:
        """Run comprehensive diagnostics on reactor consistency.
        
        Checks all applicable relations to see which are violated and identifies
        likely culprit variables causing inconsistencies. Useful for debugging
        reactor specifications.
        
        Returns:
            Dictionary containing:
            - "violated_relations": List of relation names that are violated
            - "likely_culprits": Variables most likely causing violations
            - "variable_issues": Variables with missing or problematic values
            
        Example:
            >>> reactor = Reactor.from_yaml('reactors/ARC_2015/reactor.yaml')
            >>> diag = reactor.diagnose()
            >>> print(f"Violated: {len(diag['violated_relations'])} relations")
        """
        # Create a RelationSystem in "check" mode (no modifications)
        variable_names = self.variables_dict.keys()
        variable_methods = [var.method for var in self.variables_dict.values() if var.method]
        system = RelationSystem(
            get_filtered_relations(
                self.tags,
                variable_names,
                variable_methods,
                extra_relations=self.default_relations,
            ),
            list(self.variables_dict.values()),
            mode="check",
        )
        
        # Run diagnostics through one consolidated RelationSystem API.
        diagnostics = system.diagnose()
        return {
            "violated_relations": diagnostics["violated_relations"],
            "likely_culprits": diagnostics["likely_culprits"],
            "variable_issues": diagnostics["variable_issues"],
            "soft_constraint_violations": diagnostics["soft_constraint_violations"],
            "relation_status": diagnostics["relation_status"],
        }

    def popcon(
        self,
        scan_axes: dict[str, Sequence[float]],
        *,
        outputs: Iterable[str] | None = None,
        constraints: Iterable[str] | None = None,
        constraint_tags: Iterable[str] | None = ("constraint",),
        exclude_constraints: Iterable[str] | None = None,
        where: dict[str, tuple[float | None, float | None]] | None = None,
        chunk_size: int | None = None,
    ) -> dict[str, object]:
        """Evaluate a POPCON-style scan over one or more axes.

        This method always uses the dense matrix path of `RelationSystem.evaluate`.

        Args:
            scan_axes: Mapping of variable name -> 1D sequence of scan values.
            outputs: Optional output variable names to return.
            constraints: Explicit constraint relation names/outputs.
            constraint_tags: Relation tags used to auto-select constraints.
            exclude_constraints: Relation names/outputs to exclude.
            where: Optional thresholds {name: (min, max)}.
            chunk_size: Optional row-chunk size passed to `RelationSystem.evaluate`.

        Returns:
            Dict with axes, outputs, margins, allowed mask, and diagnostics.
        """
        try:
            import numpy as np
        except Exception as exc:
            raise ImportError("popcon requires numpy.") from exc

        if not scan_axes:
            raise ValueError("scan_axes must contain at least one axis.")

        axis_order = list(scan_axes.keys())
        axes: dict[str, np.ndarray] = {}
        for name, values in scan_axes.items():
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1 or arr.size < 1:
                raise ValueError(f"scan_axes[{name}] must be a 1D array with at least one value.")
            var = self.variables_dict.get(name)
            if var is not None and var.ndim == 1:
                raise ValueError(f"scan axis '{name}' cannot be a profile variable.")
            axes[name] = arr

        grids = np.meshgrid(*[axes[name] for name in axis_order], indexing="ij")
        grid_shape = grids[0].shape

        base_values = {
            name: (None if var is None else var.current_value if var.current_value is not None else var.input_value)
            for name, var in self.variables_dict.items()
        }
        for name, var in self.variables_dict.items():
            if name in axis_order:
                continue
            if var.input_source is None:
                base_values[name] = None
        for name, grid in zip(axis_order, grids):
            base_values[name] = grid

        if constraints is not None:
            wanted = {str(item) for item in constraints}
            constraint_rels = [
                rel
                for rel in self.relations
                if rel.name in wanted
                or any(target in wanted for target in rel.outputs)
            ]
        else:
            tag_set = set(normalize_tags_to_tuple(constraint_tags or ()))
            constraint_rels = [
                rel for rel in self.relations if tag_set and tag_set.intersection(rel.tags)
            ]
        if exclude_constraints:
            exclude = {str(item) for item in exclude_constraints}
            constraint_rels = [
                rel
                for rel in constraint_rels
                if rel.name not in exclude
                and not set(rel.outputs).intersection(exclude)
            ]

        outputs_by_input: dict[str, set[str]] = {}
        for rel in self.relations:
            for inp in relation_input_names(rel):
                outputs_by_input.setdefault(inp, set()).update(rel.outputs)
        dependent: set[str] = set(axis_order)
        queue = list(axis_order)
        while queue:
            name = queue.pop()
            for out in outputs_by_input.get(name, ()):
                if out in dependent:
                    continue
                dependent.add(out)
                queue.append(out)

        if outputs is None:
            output_names = sorted(
                {
                    *base_values.keys(),
                    *[
                        output
                        for rel in self.relations
                        for output in rel.outputs
                    ],
                }
            )
        else:
            output_names = [str(name) for name in outputs]

        axis_set = set(axis_order)
        for name in output_names:
            if name in axis_set:
                continue
            var = self.variables_dict.get(name)
            if name in dependent or var is None or var.input_source is None:
                base_values[name] = None
        for rel in constraint_rels:
            for target in rel.outputs:
                if target not in axis_set:
                    base_values[target] = None
        if where:
            for name in where:
                if name in axis_set:
                    continue
                if name in dependent:
                    base_values[name] = None

        system = RelationSystem(
            list(self.relations),
            list(self.variables_dict.values()),
            mode=self.solver_mode,
            verbose=False,
        )
        values_out = system.evaluate(base_values, chunk_size=chunk_size)

        outputs_map = {name: values_out.get(name) for name in output_names if name in values_out}
        margins_map = {
            target: values_out.get(target)
            for rel in constraint_rels
            for target in rel.outputs
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

        if where:
            for name, (lo, hi) in where.items():
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

        return {
            "axes": axes,
            "axis_order": axis_order,
            "outputs": outputs_map,
            "margins": margins_map,
            "allowed": allowed,
            "diagnostics": diagnostics,
        }

    def plot_popcon(
        self,
        result: dict[str, object],
        *,
        x: str,
        y: str,
        fill: str,
        contours: list[str] | None = None,
        contour_levels: dict[str, list[float]] | None = None,
        contour_counts: dict[str, int] | None = None,
        constraint_contours: bool = True,
        slice: dict[str, int | float] | None = None,
        reduce: dict[str, str] | None = None,
        best: dict[str, str] | None = None,
        ax=None,
    ):
        """Plot POPCON results with masked fill and contour overlays."""
        from .plotting.popcon import plot_popcon

        return plot_popcon(
            result,
            x=x,
            y=y,
            fill=fill,
            contours=contours,
            contour_levels=contour_levels,
            contour_counts=contour_counts,
            constraint_contours=constraint_contours,
            slice=slice,
            reduce=reduce,
            best=best,
            ax=ax,
        )

    def __repr__(self) -> str:
        """Return a string representation of the reactor for display.
        
        Shows the reactor name, ID, and key metadata in a readable format.
        """
        parts = [f"Reactor(id='{self.id}'"]
        if self.name and self.name != self.id:
            parts.append(f", name='{self.name}'")
        if self.organization:
            parts.append(f", org='{self.organization}'")
        if self.year:
            parts.append(f", year={self.year}")
        parts.append(f", {len(self.variables_dict)} variables")
        parts.append(f", {len(self.relations)} relations")
        parts.append(")")
        return "".join(parts)

    def plot_cross_sections(self, *, ax=None, label: str | None = None):
        """Plot plasma cross-section using 95% flux surface geometry."""
        from .plotting.cross_sections import plot_cross_sections

        return plot_cross_sections(self, ax=ax, label=label)
