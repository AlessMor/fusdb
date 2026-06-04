"""User-facing reactor scenario object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import yaml

from .relation import constraint_from_expression
from .relationsystem import RelationSystem
from .registry import RELATIONS, TAGS, VARIABLES, RelationRegistry, TagRegistry, VariableRegistry
from .utils import parse_constraint_specs
from .variable import Variable


def _resolve_reactor_yaml(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_dir():
        path = path / "reactor.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _parse_variables(raw: Mapping[str, Any], *, grid_size: int | None, base_dir: Path) -> dict[str, Variable]:
    """Parse reactor YAML variables.

    Args:
        raw: Raw ``variables`` mapping from ``reactor.yaml``.
        grid_size: Optional shared profile size from the reactor grid block.
        base_dir: Directory used to resolve relative profile file paths.

    Returns:
        Variables keyed by canonical name.
    """
    variables: dict[str, Variable] = {}
    for raw_name, entry in raw.items():
        # Scalar shorthand keeps YAML compact: ``R: 3.3`` means value=3.3.
        if entry is None:
            entry = {}
        if not isinstance(entry, Mapping):
            entry = {"value": entry}

        spec = VARIABLES.get(str(raw_name))
        value = entry.get("value")
        size = entry.get("size", grid_size if spec.shape == 1 else None)

        # Profile files can be written either as ``file: path.csv`` or as
        # ``value: path.csv``.  Numeric strings are left to Variable coercion.
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
            value = _load_variable_file(
                Path(str(file_value)),
                base_dir=base_dir,
                delimiter=entry.get("delimiter"),
                usecols=entry.get("usecols"),
                skiprows=entry.get("skiprows", 0),
            )
            if np.asarray(value).ndim == 1:
                size = int(np.asarray(value).shape[0])

        var = Variable(
            str(raw_name),
            value=value,
            unit=entry.get("unit"),
            rel_tol=entry.get("rel_tol"),
            fixed=bool(entry.get("fixed", False)),
            size=size,
            constraints=entry.get("constraints"),
        )
        variables[var.name] = var
    return variables


def _load_variable_file(
    path_like: Path,
    *,
    base_dir: Path,
    delimiter: str | None = None,
    usecols: Any = None,
    skiprows: int = 0,
) -> np.ndarray:
    """Load one numeric variable file.

    Args:
        path_like: Absolute or reactor-relative file path.
        base_dir: Reactor directory used for relative paths.
        delimiter: Explicit delimiter, or ``None`` for auto-detection.
        usecols: Optional column selection passed to ``numpy.loadtxt``.
        skiprows: Header rows to skip.

    Returns:
        One-dimensional numeric values.  If the file has two or more columns
        and no ``usecols`` is supplied, the last column is treated as the
        profile value column.
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
    relation_order: tuple[str, ...] = field(default_factory=tuple)
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
        self.relation_order = tuple(str(name) for name in (self.relation_order or ()))
        self.variables = {var.name: var for var in self.variables.values()}
        if self.grid_size is not None:
            self.grid_size = int(self.grid_size)
            if self.grid_size <= 0:
                raise ValueError("grid.size must be positive.")

    @classmethod
    def from_yaml(cls, path_like: str | Path) -> "Reactor":
        """Load a reactor from ``reactor.yaml``."""
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
        """Add or replace one variable."""
        self.variables[var.name] = var

    def get_variable(self, name: str) -> Variable | None:
        """Return one loaded variable by name or alias."""
        try:
            canonical = VARIABLES.resolve(name)
        except Exception:
            canonical = str(name)
        return self.variables.get(canonical)

    def __getattr__(self, name: str) -> Variable:
        """Expose loaded variables through attribute access."""
        try:
            canonical = VARIABLES.resolve(name)
        except Exception as exc:
            raise AttributeError(name) from exc
        try:
            return self.variables[canonical]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def selected_relations(
        self,
        *,
        relation_registry: RelationRegistry = RELATIONS,
        variable_registry: VariableRegistry = VARIABLES,
        tag_registry: TagRegistry = TAGS,
    ):
        """Return relations active after tags/includes/excludes/defaults."""
        return relation_registry.get_filtered_relations(
            names=self.relation_include,
            tags=self.tags,
            exclude=self.relation_exclude,
            order=self.relation_order,
            variable_registry=variable_registry,
            tag_registry=tag_registry,
        )

    def to_relation_system(
        self,
        *,
        relation_registry: RelationRegistry = RELATIONS,
        variable_registry: VariableRegistry = VARIABLES,
        tag_registry: TagRegistry = TAGS,
        name: str | None = None,
    ) -> RelationSystem:
        """Build a RelationSystem after relation selection."""
        relations = list(self.selected_relations(
            relation_registry=relation_registry,
            variable_registry=variable_registry,
            tag_registry=tag_registry,
        ))
        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):
            relations.append(
                constraint_from_expression(
                    text,
                    name=f"reactor_constraint_{index}",
                    enforce=enforce,
                    source_kind="reactor",
                    source_name=self.name,
                )
            )

        variables = dict(self.variables)
        # Build the variable set after relation selection so excluded relations do not
        # force unused placeholders.
        required: set[str] = set()
        for rel in relations:
            required.update(rel.variables)
            for guard in rel.constraint_relations:
                required.update(guard.variables)
        for raw_name in sorted(required):
            name = variable_registry.resolve(raw_name) if raw_name in variable_registry else raw_name
            if name in variables:
                continue
            if name not in variable_registry:
                raise ValueError(f"Selected relations require unknown variable {name!r}.")
            spec = variable_registry.get(name)
            variables[name] = Variable(name, size=self.grid_size if spec.shape == 1 else None)

        return RelationSystem(
            variables=variables.values(),
            relations=relations,
            constraints=None,
            name=name or self.name,
            verbose=self.verbose,
            variable_registry=variable_registry,
        )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the configured mode."""
        return self.to_relation_system().run(self.mode, **kwargs)
