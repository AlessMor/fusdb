"""User-facing reactor scenario object."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .registry import RELATIONS, TAGS, VARIABLES
from .relationsystem import RelationSystem
from .variable import Variable


def _resolve_reactor_yaml(path_like: str | Path) -> Path:
    """Resolve a reactor path to a ``reactor.yaml`` file.

    Args:
        path_like: File or directory path.

    Returns:
        Concrete YAML path.
    """
    path = Path(path_like)
    if path.is_dir():
        path = path / "reactor.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_variable_file(path_like: Path, *, base_dir: Path, delimiter: str | None = None, usecols: Any = None, skiprows: int = 0) -> np.ndarray:
    """Load one numeric variable/profile file.

    Args:
        path_like: Absolute or reactor-relative file path.
        base_dir: Reactor directory.
        delimiter: Optional delimiter.
        usecols: Optional columns forwarded to ``numpy.loadtxt``.
        skiprows: Number of header rows.

    Returns:
        One-dimensional numeric array.
    """
    path = path_like.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Variable data file {str(path_like)!r} does not exist.")
    delimiters = (delimiter,) if delimiter is not None else (None, ",", ";")
    last_error: Exception | None = None
    for candidate in delimiters:
        try:
            data = np.loadtxt(path, delimiter=candidate, usecols=usecols, skiprows=int(skiprows))
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not load numeric variable data from {path.name!r}: {last_error}")
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        return np.asarray([float(array)], dtype=float)
    if array.ndim == 1:
        return array.astype(float)
    if array.ndim == 2 and usecols is None:
        return array[:, -1].astype(float)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0].astype(float)
    raise ValueError(f"Variable data file {path.name!r} produced a non-1D value array.")


def _parse_variables(raw: Mapping[str, Any], *, grid_size: int | None, base_dir: Path) -> dict[str, Variable]:
    """Parse reactor YAML variables.

    Args:
        raw: Raw ``variables`` mapping.
        grid_size: Optional default profile size.
        base_dir: Base path for relative profile files.

    Returns:
        Variables keyed by canonical name.
    """
    variables: dict[str, Variable] = {}
    for raw_name, entry in raw.items():
        if entry is None:
            entry = {}
        if not isinstance(entry, Mapping):
            entry = {"value": entry}
        spec = VARIABLES.get(str(raw_name))
        value = entry.get("value")
        size = entry.get("size", grid_size if spec.shape == 1 else None)
        file_value = entry.get("file")
        if file_value is None and isinstance(value, str) and spec.shape == 1:
            try:
                float(value.strip())
            except ValueError:
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = base_dir / candidate
                if candidate.is_file():
                    file_value = value
        if file_value is not None:
            value = _load_variable_file(Path(str(file_value)), base_dir=base_dir, delimiter=entry.get("delimiter"), usecols=entry.get("usecols"), skiprows=entry.get("skiprows", 0))
            if np.asarray(value).ndim == 1:
                size = int(np.asarray(value).shape[0])
        var = Variable(str(raw_name), value=value, unit=entry.get("unit"), rel_tol=entry.get("rel_tol"), fixed=bool(entry.get("fixed", False)), size=size, constraints=entry.get("constraints"))
        variables[var.name] = var
    return variables


@dataclass
class Reactor:
    """A reactor scenario with variables and relation-selection settings."""

    name: str
    organization: str | None = None
    country: str | None = None
    year: int | None = None
    doi: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    mode: str = "verify"
    variables: dict[str, Variable] = field(default_factory=dict)
    relation_include: tuple[str, ...] = field(default_factory=tuple)
    relation_exclude: tuple[str, ...] = field(default_factory=tuple)
    relation_order: tuple[Any, ...] = field(default_factory=tuple)
    constraints: Any = None
    grid_size: int | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        """Normalize simple user-facing fields."""
        if self.mode not in {"verify", "reconcile", "optimize", "ordered"}:
            raise ValueError(f"Unsupported reactor mode {self.mode!r}.")
        self.tags = tuple(str(tag).strip().lower() for tag in self.tags)
        self.relation_include = tuple(str(name) for name in (self.relation_include or ()))
        self.relation_exclude = tuple(str(name) for name in (self.relation_exclude or ()))
        self.relation_order = tuple(self.relation_order or ())
        self.variables = {var.name: var for var in self.variables.values()}
        if self.grid_size is not None:
            self.grid_size = int(self.grid_size)
            if self.grid_size <= 0:
                raise ValueError("grid.size must be positive.")

    @classmethod
    def from_yaml(cls, path_like: str | Path) -> "Reactor":
        """Load a reactor scenario from YAML.

        Args:
            path_like: Reactor directory or YAML file.

        Returns:
            Reactor instance.
        """
        path = _resolve_reactor_yaml(path_like)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("reactor.yaml must contain a mapping.")
        metadata = raw.get("metadata", {}) or {}
        solver_tags = raw.get("solver_tags", {}) or {}
        grid = raw.get("grid", {}) or {}
        grid_size = grid.get("size") if isinstance(grid, Mapping) else None
        relation_spec = raw.get("relations", {}) or {}
        return cls(
            name=str(metadata.get("name") or metadata.get("id") or path.parent.name),
            organization=metadata.get("organization"),
            country=metadata.get("country"),
            year=metadata.get("year"),
            doi=metadata.get("doi"),
            notes=metadata.get("notes"),
            tags=tuple(raw.get("tags", ()) or ()),
            mode=str(solver_tags.get("mode", raw.get("mode", "verify"))),
            variables=_parse_variables(raw.get("variables", {}) or {}, grid_size=grid_size, base_dir=path.parent),
            relation_include=tuple(relation_spec.get("include", ()) or ()) if isinstance(relation_spec, Mapping) else (),
            relation_exclude=tuple(relation_spec.get("exclude", ()) or ()) if isinstance(relation_spec, Mapping) else (),
            relation_order=tuple(relation_spec.get("order", ()) or ()) if isinstance(relation_spec, Mapping) else (),
            constraints=raw.get("constraints"),
            grid_size=grid_size,
            verbose=bool(solver_tags.get("verbosity", raw.get("verbose", False))),
        )

    def add_variable(self, var: Variable) -> None:
        """Add or replace one variable.

        Args:
            var: Variable to add.
        """
        self.variables[var.name] = var

    def get_variable(self, name: str) -> Variable | None:
        """Return one loaded variable by canonical name or alias.

        Args:
            name: Canonical name or alias.

        Returns:
            Variable or None.
        """
        try:
            canonical = VARIABLES.resolve(name)
        except Exception:
            canonical = str(name)
        return self.variables.get(canonical)

    def __getattr__(self, name: str) -> Variable:
        """Expose loaded variables through attribute access."""
        var = self.get_variable(name)
        if var is not None:
            return var
        raise AttributeError(name)

    def relations(self) -> tuple[Any, ...]:
        """Return post-filter relation objects.

        Returns:
            Tuple of relations selected for this reactor.
        """
        return RELATIONS.get_filtered_relations(names=self.relation_include, tags=TAGS.expand(self.tags), exclude=self.relation_exclude, order=None)

    def relation_system(self, *, targets: Iterable[str] | None = None, solve_for: Iterable[str] | None = None) -> RelationSystem:
        """Build a RelationSystem for this reactor.

        Args:
            targets: Optional target variables that anchor graph components.
            solve_for: Optional variables requested as solution outputs.

        Returns:
            RelationSystem instance.
        """
        return RelationSystem(
            [var.clone() for var in self.variables.values()],
            self.relations(),
            constraints=self.constraints,
            name=self.name,
            verbose=self.verbose,
            targets=targets,
            solve_for=solve_for,
        )

    def run(self, mode: str | None = None, **options: Any) -> dict[str, Any]:
        """Build a RelationSystem and run one mode.

        Args:
            mode: Optional mode override.
            **options: Mode-specific options.

        Returns:
            Result dictionary.
        """
        chosen = mode or self.mode
        system = self.relation_system()
        if chosen == "ordered":
            return system.ordered(order=self.relation_order or None, **options)
        return system.run(chosen, **options)

    def verify(self) -> dict[str, Any]:
        """Verify this reactor.

        Returns:
            Verification result.
        """
        return self.run("verify")

    def reconcile(self, **options: Any) -> dict[str, Any]:
        """Reconcile this reactor.

        Args:
            **options: Solver options.

        Returns:
            Reconciliation result.
        """
        return self.run("reconcile", **options)

    def optimize(self, **options: Any) -> dict[str, Any]:
        """Optimize this reactor.

        Args:
            **options: Optimization options.

        Returns:
            Optimization result.
        """
        return self.run("optimize", **options)

    def ordered(self, **options: Any) -> dict[str, Any]:
        """Execute this reactor's ordered recipe.

        Args:
            **options: Ordered-mode options.

        Returns:
            Ordered result.
        """
        return self.run("ordered", **options)
