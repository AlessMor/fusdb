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
from .relation_util import get_filtered_relations
from .logging_util import (
    make_logger,
    set_log_verbosity,
    log_message,
)
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
        self._log = make_logger(
            logger,
            self.__class__.__name__,
            verbose=self.verbose,
        )

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
        """Generate relations in the order they should be solved.
        
        If solving_order is specified, relations are yielded domain by domain
        in the specified order. Otherwise, all applicable relations are yielded
        based on reactor tags and available variables.
        
        Yields:
            Relation objects in solve order (no duplicates).
        """
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

        # No order specified - return all applicable relations (domain-prioritized)
        if not self.solving_order:
            from .registry import ALLOWED_SOLVING_ORDER

            domain_order = {normalize_tag(name): idx for idx, name in enumerate(ALLOWED_SOLVING_ORDER)}
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
        
        # Domains specified - yield relations domain by domain in order
        seen: list[Relation] = []
        for item in self.solving_order:
            if item in rel_by_name:
                rel = rel_by_name[item]
                if rel not in seen:
                    seen.append(rel)
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
                seen.append(rel)
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
        # Use provided mode/verbosity or fall back to instance defaults
        solver_mode = mode or self.solver_mode
        verbosity = self.verbose if verbose is None else verbose
        set_log_verbosity(self._log, verbose=verbosity)
        log_message(
            self._log,
            logging.INFO,
            "Solving reactor %s (%s) with mode=%s",
            self.id or "unknown",
            self.name or "unknown",
            solver_mode,
        )
        
        if not self.solving_order:
            rels = list(self.relations)
            log_message(self._log, logging.INFO, "Solving %d relations (no domains)", len(rels))
            system = RelationSystem(
                rels,
                list(self.variables_dict.values()),
                mode=solver_mode,
                verbose=verbosity,
                enforce_constraint_tags=tuple(enforce_constraint_tags or ()),
                enforce_constraint_names=tuple(enforce_constraint_names or ()),
            )
            system.solve()
            self.variables_dict.update(system.variables_dict)
            return

        variable_names = self.variables_dict.keys()
        variable_methods = [var.method for var in self.variables_dict.values() if var.method]
        rel_by_name = {rel.name: rel for rel in self._ordered_relations()}

        if len(self.solving_order) > 1:
            seen: dict[str, set[str]] = {}
            for item in self.solving_order:
                if item in rel_by_name:
                    rels = [rel_by_name[item]]
                    target = (
                        rels[0]._preferred_target
                        if rels[0]._preferred_target is not None
                        else next(iter(rels[0].numeric_functions), None)
                    )
                    outputs = {target} if target is not None else set(rels[0].variables)
                else:
                    domain_tags = (*self.tags, normalize_tag(item))
                    outputs = {
                        target
                        for rel in get_filtered_relations(
                            domain_tags,
                            variable_names,
                            variable_methods,
                            extra_relations=self.default_relations,
                        )
                        if (target := (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None))) is not None
                    }
                for other, other_outputs in seen.items():
                    if overlap := outputs & other_outputs:
                        warnings.warn(
                            f"Relation domains '{other}' and '{item}' both solve {sorted(overlap)}",
                            UserWarning,
                        )
                seen[item] = outputs

        for item in self.solving_order:
            if item in rel_by_name:
                rels = [rel_by_name[item]]
            else:
                domain_tags = (*self.tags, normalize_tag(item))
                rels = get_filtered_relations(
                    domain_tags,
                    variable_names,
                    variable_methods,
                    extra_relations=self.default_relations,
                )
            log_message(self._log, logging.INFO, "Solving domain %s with %d relations", item, len(rels))
            system = RelationSystem(
                rels,
                list(self.variables_dict.values()),
                mode=solver_mode,
                verbose=verbosity,
                enforce_constraint_tags=tuple(enforce_constraint_tags or ()),
                enforce_constraint_names=tuple(enforce_constraint_names or ()),
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
                or (
                    (target := (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None)))
                    is not None
                    and target in wanted
                )
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
                and (
                    (target := (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None)))
                    is None
                    or target not in exclude
                )
            ]

        outputs_by_input: dict[str, set[str]] = {}
        for rel in self.relations:
            target = (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None))
            if target is None:
                continue
            for inp in rel.required_inputs():
                outputs_by_input.setdefault(inp, set()).add(target)
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
                    *[target for rel in self.relations if (target := (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None))) is not None],
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
            target = (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None))
            if target is not None and target not in axis_set:
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
            if (target := (rel._preferred_target if rel._preferred_target is not None else next(iter(rel.numeric_functions), None))) is not None and target in values_out
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
        """Plot POPCON results with masked fill and contour overlays.

        Args:
            result: Output from Reactor.popcon.
            x: Name of x-axis scan variable.
            y: Name of y-axis scan variable.
            fill: Name of output variable for filled colormap.
            contours: Additional output variables to contour.
            contour_levels: Optional mapping of contour variable -> list of levels.
            contour_counts: Optional mapping of contour variable -> number of contour lines.
            constraint_contours: If True, plot margin==0 constraint contours.
            slice: Mapping of remaining axes to fixed index or value.
            reduce: {"metric": name, "mode": "max"|"min"} to reduce remaining dims.
            best: {"metric": name, "mode": "max"|"min"} for annotating best point.
            ax: Optional matplotlib Axes to plot on.

        Returns:
            Matplotlib Axes.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        axes = result.get("axes", {})
        axis_order = result.get("axis_order", list(axes.keys()))
        outputs = result.get("outputs", {})
        margins = result.get("margins", {})
        allowed = result.get("allowed")

        if x not in axes or y not in axes:
            raise ValueError("x and y must be scan axes present in result['axes'].")

        if allowed is None:
            raise ValueError("Result missing 'allowed' mask.")

        x_vals = axes[x]
        y_vals = axes[y]

        def _select_2d(arr: np.ndarray, *, label: str) -> np.ndarray:
            data = np.asarray(arr)
            if data.ndim == 0:
                return np.broadcast_to(data, (len(x_vals), len(y_vals)))
            if data.ndim == 1:
                if data.shape[0] == len(x_vals):
                    return np.broadcast_to(data[:, None], (len(x_vals), len(y_vals)))
                if data.shape[0] == len(y_vals):
                    return np.broadcast_to(data[None, :], (len(x_vals), len(y_vals)))
                raise ValueError(f"{label} length must match x or y axis for plotting.")
            if data.ndim < 2:
                raise ValueError(f"{label} must be at least 2D for plotting.")

            if data.ndim > 2:
                if slice is None and reduce is None:
                    raise ValueError("Provide slice or reduce for scans with >2 dimensions.")

                if slice is not None:
                    indexers = []
                    for name in axis_order:
                        if name in (x, y):
                            indexers.append(slice(None))
                            continue
                        if name not in slice:
                            raise ValueError(f"Missing slice for axis '{name}'.")
                        selector = slice[name]
                        axis_vals = axes[name]
                        if isinstance(selector, int):
                            indexers.append(selector)
                        else:
                            idx = int(np.argmin(np.abs(axis_vals - float(selector))))
                            indexers.append(idx)
                    data = data[tuple(indexers)]
                else:
                    metric_name = reduce.get("metric") if reduce else None
                    mode = reduce.get("mode", "max") if reduce else "max"
                    metric = outputs.get(metric_name) if metric_name else None
                    if metric is None:
                        raise ValueError("reduce requires a metric present in outputs.")

                    perm = [axis_order.index(x), axis_order.index(y)]
                    perm += [axis_order.index(name) for name in axis_order if name not in (x, y)]
                    data_r = np.moveaxis(data, perm, range(len(perm)))
                    metric_r = np.moveaxis(np.asarray(metric), perm, range(len(perm)))
                    allowed_r = np.moveaxis(np.asarray(allowed), perm, range(len(perm)))

                    flat_data = data_r.reshape(data_r.shape[0], data_r.shape[1], -1)
                    flat_metric = metric_r.reshape(metric_r.shape[0], metric_r.shape[1], -1)
                    flat_allowed = allowed_r.reshape(allowed_r.shape[0], allowed_r.shape[1], -1)

                    if mode == "min":
                        masked = np.where(flat_allowed, flat_metric, np.inf)
                        best_idx = np.nanargmin(masked, axis=2)
                    else:
                        masked = np.where(flat_allowed, flat_metric, -np.inf)
                        best_idx = np.nanargmax(masked, axis=2)

                    data = np.take_along_axis(flat_data, best_idx[..., None], axis=2)[..., 0]

            remaining = [name for name in axis_order if name in (x, y)]
            if remaining != [x, y]:
                data = np.moveaxis(data, [remaining.index(x), remaining.index(y)], [0, 1])

            return data

        def _as_float_array(arr: np.ndarray) -> np.ndarray:
            if arr.dtype != object:
                return arr
            out = np.empty(arr.shape, dtype=float)
            it = np.nditer(arr, flags=["multi_index", "refs_ok"])
            for item in it:
                try:
                    out[it.multi_index] = float(item.item())
                except Exception:
                    out[it.multi_index] = np.nan
            return out

        fill_data = outputs.get(fill)
        if fill_data is None:
            raise ValueError(f"Fill variable '{fill}' not found in outputs.")

        fill_2d = _select_2d(_as_float_array(np.asarray(fill_data)), label=fill)
        allowed_2d = _select_2d(np.asarray(allowed), label="allowed")

        masked_fill = np.ma.masked_where(~allowed_2d, fill_2d)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        def _plot_ready(data: np.ndarray) -> np.ndarray:
            arr = np.asarray(data)
            if arr.shape == (len(x_vals), len(y_vals)):
                return arr.T
            return arr

        mesh = ax.pcolormesh(x_vals, y_vals, _plot_ready(masked_fill), shading="auto", cmap="viridis")
        plt.colorbar(mesh, ax=ax, label=fill)

        legend_handles = []
        legend_labels = []
        pretty_names = {"Q_sci": "Q", "n_GW": "n_greenwald"}

        contour_colors = [
            "#d62728",  # red
            "#ff7f0e",  # orange
            "#8c564b",  # brown
            "#e377c2",  # magenta
            "#7f7f7f",  # gray
            "#a52a2a",  # dark red
            "#4d4d4d",  # dark gray
            "#b15928",  # dark orange/brown
        ]
        color_cycle = contour_colors
        color_idx = 0

        def _next_color() -> tuple[float, float, float]:
            nonlocal color_idx
            color = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1
            return color

        def _label_values(cs):
            if not cs.levels.size:
                return
            ax.clabel(
                cs,
                fmt=lambda val: f"{val:g}",
                inline=True,
                fontsize=8,
            )

        if constraint_contours:
            for name, margin in margins.items():
                if margin is None:
                    continue
                margin_2d = _select_2d(_as_float_array(np.asarray(margin)), label=name)
                color = _next_color()
                cs = ax.contour(
                    x_vals,
                    y_vals,
                    _plot_ready(margin_2d),
                    levels=[0.0],
                    colors=[color],
                    linewidths=1.0,
                )
                _label_values(cs)
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([], [], color=color, linewidth=1.0))
                legend_labels.append(pretty_names.get(name, name))

        if contours:
            for name in contours:
                data = outputs.get(name)
                if data is None:
                    continue
                if name in ("n_GW", "n_greenwald"):
                    try:
                        val = float(np.asarray(data, dtype=float).reshape(-1)[0])
                    except Exception:
                        continue
                    color = _next_color()
                    from matplotlib.lines import Line2D
                    if y in ("n_avg", "n_e"):
                        ax.axhline(val, color=color, linewidth=1.2)
                        ax.text(x_vals[len(x_vals) // 2], val, f"{val:g}", color=color, fontsize=8, va="bottom")
                    elif x in ("n_avg", "n_e"):
                        ax.axvline(val, color=color, linewidth=1.2)
                        ax.text(val, y_vals[len(y_vals) // 2], f"{val:g}", color=color, fontsize=8, ha="left")
                    legend_handles.append(Line2D([], [], color=color, linewidth=1.2))
                    legend_labels.append(pretty_names.get(name, name))
                    continue
                data_2d = _select_2d(_as_float_array(np.asarray(data)), label=name)
                color = _next_color()
                levels = None
                if contour_levels and name in contour_levels:
                    levels = contour_levels[name]
                elif contour_counts and name in contour_counts:
                    try:
                        levels = int(contour_counts[name])
                    except Exception:
                        levels = None
                finite = np.isfinite(data_2d)
                is_constant = False
                if finite.any():
                    vmin = float(np.nanmin(data_2d))
                    vmax = float(np.nanmax(data_2d))
                    scale = max(abs(vmin), abs(vmax), 1.0)
                    is_constant = abs(vmax - vmin) <= 1e-12 * scale
                if levels is None and is_constant and fill_2d is not None:
                    const_val = float(np.nanmean(data_2d))
                    cs = ax.contour(
                        x_vals,
                        y_vals,
                        _plot_ready(fill_2d),
                        levels=[const_val],
                        colors=[color],
                        linewidths=0.9,
                        alpha=0.9,
                    )
                    _label_values(cs)
                    from matplotlib.lines import Line2D
                    legend_handles.append(Line2D([], [], color=color, linewidth=1.0))
                    legend_labels.append(pretty_names.get(name, name))
                    continue
                cs = ax.contour(
                    x_vals,
                    y_vals,
                    _plot_ready(data_2d),
                    levels=levels,
                    colors=[color],
                    linewidths=0.9,
                    alpha=0.9,
                )
                _label_values(cs)
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([], [], color=color, linewidth=1.0))
                legend_labels.append(pretty_names.get(name, name))

        if best:
            metric_name = best.get("metric")
            mode = best.get("mode", "max")
            metric = outputs.get(metric_name)
            if metric is not None:
                metric_2d = _select_2d(_as_float_array(np.asarray(metric)), label=metric_name)
                metric_masked = np.where(allowed_2d, metric_2d, np.nan)
                if np.all(np.isnan(metric_masked)):
                    return ax
                if mode == "min":
                    idx = np.nanargmin(metric_masked)
                else:
                    idx = np.nanargmax(metric_masked)
                ix, iy = np.unravel_index(idx, metric_masked.shape)
                ax.scatter([x_vals[ix]], [y_vals[iy]], color="red", s=30, zorder=5)
                ax.annotate(
                    f"{metric_name}={metric_2d[ix, iy]:.3g}",
                    (x_vals[ix], y_vals[iy]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    color="red",
                )

        if legend_handles:
            ax.legend(legend_handles, legend_labels, title="Contours", loc="best", fontsize=8)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"POPCON: {fill}")
        ax.grid(True, alpha=0.3)
        return ax

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
        """Plot plasma cross-section using 95% flux surface geometry.
        
        Args: ax (matplotlib Axes), label (str). Returns: matplotlib Axes.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        values = {}
        for name in ("R", "a", "kappa_95", "delta_95"):
            var = self.variables_dict.get(name)
            values[name] = None if var is None else var.current_value if var.current_value is not None else var.input_value
        R, a, kappa_95, delta_95 = (
            values["R"],
            values["a"],
            values["kappa_95"],
            values["delta_95"],
        )
        
        if any(val is None for val in (R, a, kappa_95, delta_95)):
            missing = []
            if R is None: missing.append("R")
            if a is None: missing.append("a")
            if kappa_95 is None: missing.append("kappa_95")
            if delta_95 is None: missing.append("delta_95")
            raise ValueError(f"Missing 95% geometry variables for plotting: {', '.join(missing)}")
        
        Rv, av, kv, dv = R, a, kappa_95, delta_95
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        theta = np.linspace(0, 2 * np.pi, 200)
        r_vals = Rv + av * np.cos(theta + dv * np.sin(theta))
        z_vals = kv * av * np.sin(theta)
        ax.plot(r_vals, z_vals, label=label or self.name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.grid(True, alpha=0.3)
        return ax
