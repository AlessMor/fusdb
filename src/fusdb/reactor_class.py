"""User-facing reactor container and thin orchestration wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import logging

import yaml

from .registry import parse_variables, validate_solver_tags
from .relation_class import Relation
from .popcon_class import Popcon
from .relationsystem_class import RelationSystem
from .utils import normalize_country, normalize_tags_to_tuple
from .variable_class import Variable

logger = logging.getLogger(__name__)


@dataclass
class Reactor:
    """Container for reactor metadata, variables, and applicable relations.

    Attributes:
        path: Path to source reactor YAML file.
        id: Reactor identifier.
        name: Human-readable reactor name.
        organization: Reactor organization metadata.
        country: Reactor country metadata.
        year: Reactor design/publication year.
        doi: Reactor DOI reference.
        notes: Optional free-text notes.
        tags: Reactor tags used for relation filtering.
        solving_order: Optional ordered domain/relation solve list copied from YAML.
        solver_mode: Solver mode (``"overwrite"`` or ``"check"``).
        verbose: Runtime verbosity flag.
        relations: Optional caller-provided relation list.
        default_relations: Default/assumption relations generated at load.
        variables_dict: Reactor variables keyed by name.
        last_solve_stop_reason: Last RelationSystem stop reason from ``solve()``.
        last_solve_final_check: Last RelationSystem terminal final-check summary.
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
    last_solve_stop_reason: str | None = None
    last_solve_final_check: dict[str, object] = field(default_factory=dict)
    _log: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize per-instance logger.

        Args:
            None.

        Returns:
            None.
        """
        # Build one child logger and align level with verbose flag.
        self._log = logger.getChild(self.__class__.__name__)
        self._log.setLevel(logging.INFO if self.verbose else logging.WARNING)

    @classmethod
    def from_yaml(cls, path_like: str | Path) -> "Reactor":
        """Create one reactor object from YAML.

        Args:
            path_like: Reactor YAML path, directory, or reactor-like path.

        Returns:
            Initialized Reactor object.
        """
        # Resolve one concrete reactor.yaml path from flexible user input.
        path = Path(path_like).expanduser()
        if not path.is_file():
            if path.is_dir() and (path / "reactor.yaml").is_file():
                path = path / "reactor.yaml"
            else:
                start = Path.cwd()
                root = start
                for parent in (start, *start.parents):
                    if (parent / "reactors").is_dir() and (parent / "src" / "fusdb").is_dir():
                        root = parent
                        break
                candidate = root / path
                if candidate.is_file():
                    path = candidate
                elif candidate.is_dir() and (candidate / "reactor.yaml").is_file():
                    path = candidate / "reactor.yaml"
                elif (root / "reactors" / path).is_dir():
                    path = root / "reactors" / path / "reactor.yaml"
                elif (root / "reactors" / f"{path}.yaml").is_file():
                    path = root / "reactors" / f"{path}.yaml"

        # Load one YAML document and parse metadata/solver settings.
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        metadata = raw.get("metadata", {}) or {}
        reactor_id = metadata.get("id") or path.parent.name
        reactor_name = metadata.get("name") or reactor_id

        tags = normalize_tags_to_tuple(raw.get("tags", []) or ())
        solver_tags = raw.get("solver_tags", {}) or {}
        if "abs_tol" in solver_tags:
            raise ValueError("solver_tags.abs_tol is unsupported. Use only rel_tol.")
        # Use the registry's canonical solver mode names directly.
        raw_solver_mode = solver_tags.get("mode", "overwrite")
        solver_mode = "overwrite" if raw_solver_mode is None else str(raw_solver_mode)
        default_rel_tol = solver_tags.get("rel_tol")
        solver_tags_for_validation = dict(solver_tags)
        if "mode" in solver_tags_for_validation:
            solver_tags_for_validation["mode"] = solver_mode
        validate_solver_tags(solver_tags_for_validation, log=logger)
        verbose = bool(solver_tags.get("verbosity", False))
        solving_order = list(solver_tags.get("solving_order", []) or ())

        # Build variable objects and apply solver-level tolerance defaults.
        variables_dict = parse_variables(raw.get("variables", {}))
        for var in variables_dict.values():
            if default_rel_tol is not None and var.rel_tol is None:
                var.rel_tol = float(default_rel_tol)

        # Attach default/fallback relations; defaults errors are fatal by design.
        from .registry.reactor_defaults import apply_reactor_defaults

        default_relations: list[Relation] = apply_reactor_defaults(variables_dict)

        # Build one Reactor object; relation filtering/ordering belongs to RelationSystem.
        return cls(
            path=path,
            id=reactor_id,
            name=reactor_name,
            organization=metadata.get("organization"),
            country=normalize_country(metadata.get("country")),
            year=metadata.get("year"),
            doi=metadata.get("doi"),
            notes=metadata.get("notes"),
            tags=tags,
            solving_order=solving_order,
            solver_mode=solver_mode,
            verbose=verbose,
            variables_dict=variables_dict,
            default_relations=default_relations,
        )

    def make_relationsystem(
        self,
        *,
        mode: str | None = None,
        verbose: bool | None = None,
    ) -> RelationSystem:
        """Build one runtime relation system bound to this reactor values.

        Args:
            mode: Optional solver mode override.
            verbose: Optional verbosity override.

        Returns:
            Reactor-bound RelationSystem instance.
        """
        # Pass raw reactor settings through; RelationSystem builds the ordered graph.
        variable_methods = [var.method for var in self.variables_dict.values() if var.method]
        return RelationSystem(
            relations=list(self.relations),
            variables=list(self.variables_dict.values()),
            mode=mode or self.solver_mode,
            verbose=self.verbose if verbose is None else bool(verbose),
            reactor_tags=tuple(self.tags),
            solving_order=tuple(self.solving_order),
            variable_methods=tuple(variable_methods),
            default_relations=list(self.default_relations),
        )

    def solve(
        self,
        mode: str | None = None,
        *,
        verbose: bool | None = None,
    ) -> None:
        """Solve for unknown reactor values in-place.

        Args:
            mode: Optional solver mode override.
            verbose: Optional verbosity override.

        Returns:
            None.
        """
        # Solve through one runtime RelationSystem and copy values back.
        system = self.make_relationsystem(
            mode=mode,
            verbose=verbose,
        )
        system.solve()
        self.variables_dict.update(system.variables_dict)
        stop_reason = system.last_result.get("stop_reason")
        self.last_solve_stop_reason = None if stop_reason is None else str(stop_reason)
        self.last_solve_final_check = dict(system.last_result.get("final_check") or {})

    @property
    def solve_status(self) -> str:
        """Return one compact status string for the latest ``solve()`` run.

        Args:
            None.

        Returns:
            Readable status combining stop reason and final-check summary.
        """
        # Report unsolved reactors explicitly for table views.
        if self.last_solve_stop_reason is None:
            return "not_solved"

        # Keep one compact summary that includes terminal final-check counts.
        final_check = self.last_solve_final_check or {}
        violated_count = int(final_check.get("violated_count", 0) or 0)
        undecidable_count = int(final_check.get("undecidable_count", 0) or 0)
        if bool(final_check.get("all_satisfied", False)):
            return f"{self.last_solve_stop_reason}; final_check=all_satisfied"
        if violated_count and undecidable_count:
            return (
                f"{self.last_solve_stop_reason}; "
                f"final_check={violated_count} violated, {undecidable_count} undecidable"
            )
        if violated_count:
            return f"{self.last_solve_stop_reason}; final_check={violated_count} violated"
        if undecidable_count:
            return f"{self.last_solve_stop_reason}; final_check={undecidable_count} undecidable"
        return self.last_solve_stop_reason

    def diagnose(self) -> dict[str, object]:
        """Return consolidated diagnostics for current reactor values.

        Args:
            None.

        Returns:
            Diagnostics dictionary with violated relations and likely culprits.
        """
        # Diagnose through one check-mode runtime RelationSystem.
        system = self.make_relationsystem(
            mode="check",
            verbose=False,
        )
        return system.diagnose()

    def to_table_payload(
        self,
        *,
        reactor_id: str | None = None,
        metadata_fields: Sequence[str] | None = None,
        include_diagnostics: bool = True,
    ) -> dict[str, object]:
        """Return one normalized payload for multi-reactor table rendering.

        Args:
            reactor_id: Optional override for the column identifier.
            metadata_fields: Optional metadata field names to include.
            include_diagnostics: When ``True``, include ``diagnose()`` status data.

        Returns:
            Mapping with ``reactor_id``, ``metadata``, ``variables``, and ``diagnostics``.
        """
        # Resolve one stable identifier for table columns.
        rid = reactor_id or self.id or self.name or "reactor"
        fields = tuple(
            metadata_fields
            or (
                "id",
                "name",
                "organization",
                "country",
                "tags",
                "solve_status",
                "year",
                "doi",
                "notes",
            )
        )

        # Extract requested metadata values in one compact mapping.
        metadata = {field: getattr(self, field, None) for field in fields}

        # Optionally compute diagnostics used by table status styling.
        diagnostics = self.diagnose() if include_diagnostics else {}
        variable_issue_map: dict[str, tuple[str, int | None]] = {}
        for name, status, rank in diagnostics.get("variable_issues", ()):
            variable_issue_map[str(name)] = (str(status), rank)
        # Build one per-variable payload without mutating reactor state.
        variables_payload: dict[str, dict[str, Any]] = {}
        for name, var in self.variables_dict.items():
            input_value = var.input_value
            current_value = var.current_value if var.current_value is not None else input_value
            status, rank = variable_issue_map.get(name, (None, None))
            variables_payload[name] = {
                "input_value": input_value,
                "current_value": current_value,
                "input_source": var.input_source,
                "rel_tol": var.rel_tol,
                "diag_status": status,
                "diag_rank": rank,
            }

        return {
            "reactor_id": rid,
            "metadata": metadata,
            "variables": variables_payload,
            "diagnostics": diagnostics,
        }

    def popcon(
        self,
        scan_axes: dict[str, Sequence[float]],
        *,
        outputs: Iterable[str] | None = None,
        constraints: Iterable[str] | None = None,
        exclude_constraints: Iterable[str] | None = None,
        where: dict[str, tuple[float | None, float | None]] | None = None,
        chunk_size: int | None = None,
    ) -> Popcon:
        """Run one POPCON scan and return an already-executed Popcon object.

        Args:
            scan_axes: Scan axis mapping of variable name to 1D values.
            outputs: Optional outputs to include.
            constraints: Optional explicit constraint relation/output selectors.
            exclude_constraints: Optional constraints to exclude.
            where: Optional numeric thresholds for additional filtering.
            chunk_size: Optional dense-evaluation chunk size.

        Returns:
            Executed Popcon object including outputs, margins, allowed mask, and diagnostics.
        """
        return Popcon(
            reactor=self,
            scan_axes=scan_axes,
            outputs=outputs,
            constraints=constraints,
            exclude_constraints=exclude_constraints,
            where=where,
            chunk_size=chunk_size,
        )

    def plot_popcon(
        self,
        result: dict[str, object] | Popcon,
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
        """Plot POPCON scan results.

        Args:
            result: POPCON result mapping or :class:`~fusdb.popcon_class.Popcon`.
            x: X-axis variable name.
            y: Y-axis variable name.
            fill: Fill variable name.
            contours: Optional additional contour outputs.
            contour_levels: Optional contour level mapping.
            contour_counts: Optional contour count mapping.
            constraint_contours: Include constraint contours when ``True``.
            slice: Optional axis slicing for >2D results.
            reduce: Optional reduction config for >2D results.
            best: Optional best-point marker config.
            ax: Optional matplotlib axis.

        Returns:
            Matplotlib axis with rendered POPCON plot.
        """
        from .plotting.popcon import plot_popcon

        result_payload = result.to_result() if isinstance(result, Popcon) else result
        # Delegate visualization to plotting helper module.
        return plot_popcon(
            result_payload,
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

    def plot_cross_sections(self, *, ax=None, label: str | None = None):
        """Plot reactor plasma cross-sections.

        Args:
            ax: Optional matplotlib axis.
            label: Optional plot label.

        Returns:
            Matplotlib axis with rendered cross-section plot.
        """
        from .plotting.cross_sections import plot_cross_sections

        # Delegate visualization to plotting helper module.
        return plot_cross_sections(self, ax=ax, label=label)

    def __repr__(self) -> str:
        """Return a compact reactor representation string.

        Args:
            None.

        Returns:
            Printable reactor summary string.
        """
        # Build a stable summary of key reactor metadata and model size.
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
