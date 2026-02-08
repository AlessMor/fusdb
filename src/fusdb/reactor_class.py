"""Reactor class for loading and solving fusion reactor design specifications.

This module provides the Reactor class which loads reactor specifications from
YAML files, manages variables, filters applicable relations, and orchestrates
the solving process.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable
import logging
import warnings
from .relation_class import Relation
from .relationsystem_class import RelationSystem
from .relation_util import get_filtered_relations
from .registry import parse_variables, validate_solver_tags
from .utils import normalize_tag, normalize_tags_to_tuple, load_yaml, normalize_country, normalize_solver_mode
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
    _log: logging.LoggerAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize per-instance logger with context."""
        base_logger = logger.getChild(self.__class__.__name__)
        base_logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        self._log = logging.LoggerAdapter(
            base_logger,
            {"reactor_id": self.id, "name": self.name, "mode": self.solver_mode},
        )

    @staticmethod
    def _resolve_reactor_yaml(path_like: str | Path) -> Path:
        """Resolve a path or name to a reactor.yaml file.
        
        Args:
            path_like: Path, directory, or reactor name.
        
        Returns:
            Resolved Path to reactor.yaml file.
        """
        path = Path(path_like).expanduser()
        if path.is_file():
            return path
        if path.is_dir():
            candidate = path / "reactor.yaml"
            if candidate.is_file():
                return candidate
        
        # Find repo root by walking up from cwd
        start = Path.cwd()
        root = start
        for parent in (start, *start.parents):
            if (parent / "reactors").is_dir() and (parent / "src" / "fusdb").is_dir():
                root = parent
                break
        
        candidate = root / path
        if candidate.is_file():
            return candidate
        if candidate.is_dir() and (candidate / "reactor.yaml").is_file():
            return candidate / "reactor.yaml"
        reactors_dir = root / "reactors"
        if (reactors_dir / path).is_dir():
            return reactors_dir / path / "reactor.yaml"
        if (reactors_dir / f"{path}.yaml").is_file():
            return reactors_dir / f"{path}.yaml"
        return path

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
        # Resolve path and load YAML content
        path = cls._resolve_reactor_yaml(path_like)
        raw = load_yaml(path)
        
        # Extract and normalize metadata fields
        metadata = raw.get("metadata", {}) or {}
        reactor_id = metadata.get("id") or path.parent.name
        reactor_name = metadata.get("name") or reactor_id
        
        # Parse reactor tags and solver configuration
        tags = normalize_tags_to_tuple(raw.get("tags", []) or ())
        solver_tags = raw.get("solver_tags", {}) or {}
        solver_mode = normalize_solver_mode(solver_tags.get("mode", "overwrite"))
        solver_tags_for_validation = dict(solver_tags)
        if "mode" in solver_tags_for_validation:
            solver_tags_for_validation["mode"] = solver_mode
        validate_solver_tags(solver_tags_for_validation, log=logger)
        verbose = bool(solver_tags.get("verbosity", False))
        solving_order = list(solver_tags.get("solving_order", []) or ())
        
        # Parse variables from YAML and create Variable objects
        variables_dict = parse_variables(raw.get("variables", {}))
        
        # Apply default values for common variables (e.g., constants)
        default_relations: list[Relation] = []
        try:
            from .registry.reactor_defaults import apply_reactor_defaults
            default_relations = apply_reactor_defaults(variables_dict)
        except Exception:
            pass  # Non-fatal if defaults can't be applied
        
        # Create the Reactor instance with all parsed data
        reactor = cls(
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
        
        # Filter and load applicable relations based on reactor tags
        # (Relations are auto-loaded by get_filtered_relations)
        
        # Build filtered relation list (relations auto-loaded by get_filtered_relations)
        reactor.relations = list(reactor._ordered_relations())
        
        return reactor

    def _relation_filter_inputs(self) -> tuple[Iterable[str], list[str]]:
        """Return variable names and method override relation names for filtering."""
        variable_names = self.variables_dict.keys()
        variable_methods = [var.method for var in self.variables_dict.values() if var.method]
        return variable_names, variable_methods

    def _ordered_relations(self) -> Iterable[Relation]:
        """Generate relations in the order they should be solved.
        
        If solving_order is specified, relations are yielded domain by domain
        in the specified order. Otherwise, all applicable relations are yielded
        based on reactor tags and available variables.
        
        Yields:
            Relation objects in solve order (no duplicates).
        """
        variable_names, variable_methods = self._relation_filter_inputs()

        base_relations = list(
            get_filtered_relations(
                self.tags,
                variable_names,
                variable_methods,
                extra_relations=self.default_relations,
            )
        )
        rel_by_name = {rel.name: rel for rel in base_relations}

        # No order specified - return all applicable relations
        if not self.solving_order:
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

    def solve(self, mode: str | None = None, *, verbose: bool | None = None) -> None:
        """Solve for unknown reactor variables using physics relations.
        
        Executes the iterative solver to compute unknown variables from known ones.
        If solving_order is specified, solves each domain in sequence.
        
        Args:
            mode: Override solver mode ("overwrite" or "check").
                 If None, uses self.solver_mode.
            verbose: Override verbosity setting. If None, uses self.verbose.
        Example:
            >>> reactor = Reactor.from_yaml('reactors/ARC_2015/reactor.yaml')
            >>> reactor.solve(verbose=True)  # Show detailed solving progress
        """
        # Use provided mode/verbosity or fall back to instance defaults
        solver_mode = mode or self.solver_mode
        verbosity = self.verbose if verbose is None else verbose
        self._log.logger.setLevel(logging.INFO if verbosity else logging.WARNING)
        self._log.extra["mode"] = solver_mode
        self._log.info(
            "Solving reactor %s (%s) with mode=%s",
            self.id or "unknown",
            self.name or "unknown",
            solver_mode,
        )
        
        if not self.solving_order:
            rels = list(self.relations)
            self._log.info("Solving %d relations (no domains)", len(rels))
            system = RelationSystem(
                rels,
                list(self.variables_dict.values()),
                mode=solver_mode,
                verbose=verbosity,
            )
            system.solve()
            self.variables_dict.update(system.variables_dict)
            return

        variable_names, variable_methods = self._relation_filter_inputs()
        rel_by_name = {rel.name: rel for rel in self._ordered_relations()}

        if len(self.solving_order) > 1:
            seen: dict[str, set[str]] = {}
            for item in self.solving_order:
                if item in rel_by_name:
                    rels = [rel_by_name[item]]
                    outputs = {rels[0].output}
                else:
                    domain_tags = (*self.tags, normalize_tag(item))
                    outputs = {
                        rel.output
                        for rel in get_filtered_relations(
                            domain_tags,
                            variable_names,
                            variable_methods,
                            extra_relations=self.default_relations,
                        )
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
            self._log.info("Solving domain %s with %d relations", item, len(rels))
            system = RelationSystem(
                rels,
                list(self.variables_dict.values()),
                mode=solver_mode,
                verbose=verbosity,
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
        system = RelationSystem(
            get_filtered_relations(
                self.tags,
                *self._relation_filter_inputs(),
                extra_relations=self.default_relations,
            ),
            list(self.variables_dict.values()),
            mode="check",
        )
        
        # Run diagnostics
        violated, culprits = system.diagnose_relations(return_culprits=True)
        
        return {
            "violated_relations": violated,
            "likely_culprits": culprits,
            "variable_issues": system.diagnose_variables(),
        }

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

        R = Variable.get_from_dict(self.variables_dict, "R", allow_override=True)
        a = Variable.get_from_dict(self.variables_dict, "a", allow_override=True)
        kappa_95 = Variable.get_from_dict(self.variables_dict, "kappa_95", allow_override=True)
        delta_95 = Variable.get_from_dict(self.variables_dict, "delta_95", allow_override=True)
        
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
