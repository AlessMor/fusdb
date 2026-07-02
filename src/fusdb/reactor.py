"""User-facing reactor scenario object."""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

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


def _format_table_value(value: Any) -> str:
    """Compact scalar/profile formatting for HTML table cells."""
    if value is None:
        return ""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if array.ndim == 0 or array.size == 1:
        scalar = float(array.ravel()[0])
        if scalar == 0:
            return "0"
        if abs(scalar) >= 1e4 or abs(scalar) < 1e-3:
            return f"{scalar:.3e}"
        return f"{scalar:.4g}"
    return f"prof[{array.size}] mean={np.nanmean(array):.3g}"


def _table_cell_display(var: Variable | None, used: bool) -> tuple[str, str, str]:
    """Return ``(background, foreground, html_text)`` for one variable cell."""
    background = ""
    color = "#000000"
    text = ""
    if var is None:
        return background, color, text

    has_input = var.input_value is not None
    input_value = var.input_value
    value = var.value
    if has_input and used and value is not None:
        try:
            input_array = np.asarray(input_value, dtype=float)
            value_array = np.asarray(value, dtype=float)
            exact = bool(np.array_equal(input_array, value_array))
            scale = max(
                float(np.max(np.abs(input_array))),
                float(np.max(np.abs(value_array))),
                1e-300,
            )
            tolerance = max(
                float(var.abs_tol or 0.0),
                float(var.rel_tol or 0.0) * scale,
            )
            within = bool(np.all(np.abs(value_array - input_array) <= tolerance))
        except Exception:
            exact, within = False, False
        if exact:
            background, color, text = "#c6efce", "#006100", _format_table_value(value)
        elif within:
            background, color = "#ffeb9c", "#9c6500"
            text = f"{_format_table_value(input_value)} ({_format_table_value(value)})"
        else:
            background, color = "#ffc7ce", "#9c0006"
            text = f"<b>{_format_table_value(input_value)}</b> &rarr; {_format_table_value(value)}"
    elif has_input and not used:
        color, text = "#6E6E6E", _format_table_value(value if value is not None else input_value)
    elif (not has_input) and used and value is not None:
        color, text = "#FFFFFF", _format_table_value(value)
    elif has_input and used and value is None:
        color, text = "#606060", _format_table_value(input_value)
    return background, color, text


def _sort_table_variable_names(names: Iterable[str]) -> tuple[str, ...]:
    """Sort variable names by registry order, then alphabetically."""
    registry_order = {
        spec.name: index
        for index, spec in enumerate(VARIABLES)
    }
    return tuple(
        sorted(
            names,
            key=lambda name: (registry_order.get(name, len(registry_order)), name),
        )
    )


class SolvedColumn(NamedTuple):
    """One table column's display data, extracted from a reactor or system.

    Picklable, so it doubles as the result a worker process returns from a
    parallel solve (see :func:`solve_reactors`).
    """

    name: str
    variables_by_name: Mapping[str, Variable]
    active_variable_names: frozenset[str]
    relation_names_by_variable: Mapping[str, tuple[str, ...]]
    result: Mapping[str, Any]


def _table_column(source: Any) -> SolvedColumn:
    """Extract a :class:`SolvedColumn` from a reactor, system, or column.

    A :class:`RelationSystem` contributes active variables, per-variable relation
    names (for cell tooltips), and the result of its most recent run (for header
    colouring). A :class:`Reactor` contributes only its current variable values.
    An already-built :class:`SolvedColumn` is returned unchanged.
    """
    if isinstance(source, SolvedColumn):
        return source
    if hasattr(source, "variables_by_name"):  # RelationSystem
        relations: dict[str, list[str]] = {}
        for rel in getattr(source, "relations", ()):
            for variable_name in rel.variables:
                relations.setdefault(variable_name, []).append(rel.name)
        return SolvedColumn(
            source.name,
            source.variables_by_name,
            frozenset(getattr(source, "active_variable_names", ())),
            {name: tuple(dict.fromkeys(names)) for name, names in relations.items()},
            getattr(source, "last_result", None) or {},
        )
    return SolvedColumn(source.name, source.variables, frozenset(), {}, {})


def _displayed_variable_names(columns: Iterable[SolvedColumn], variable_names: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve the row order/subset: the explicit list, or active + supplied."""
    if variable_names is not None:
        return tuple(variable_names)
    names: set[str] = set()
    for column in columns:
        names.update(column.active_variable_names)
        names.update(name for name, var in column.variables_by_name.items() if var.input_value is not None)
    return _sort_table_variable_names(names)


def variables_table(*sources: Any, variable_names: Iterable[str] | None = None) -> str:
    """Render current variable values for one or more reactors/systems as HTML.

    Each positional source is a :class:`Reactor`, a :class:`RelationSystem`, or a
    :class:`SolvedColumn` (e.g. from :func:`solve_reactors`); columns are sources
    and rows are variables. Reactor columns show current values; solved systems
    and columns additionally highlight active variables, colour input->output
    changes, add relation tooltips, and colour the header by solve success.
    ``variable_names`` overrides the row order/subset; when omitted, all active
    and user-supplied variables are shown.

    Returns:
        HTML ``<table>`` string.
    """
    columns = [_table_column(source) for source in sources]
    ordered_names = _displayed_variable_names(columns, variable_names)

    parts = ["<table style='border-collapse:collapse;font-size:0.8em'>"]
    parts.append("<tr><th style='text-align:left;padding:2px 8px'>variable</th>")
    for column in columns:
        style = "padding:2px 8px"
        if column.result:
            style += f";color:{'#1EFF00' if column.result.get('success') else '#c00000'}"
        parts.append(f"<th style='{style}'>{html.escape(column.name)}</th>")
    parts.append("</tr>")

    for name in ordered_names:
        parts.append(
            f"<tr><td style='text-align:left;padding:2px 8px;font-weight:bold'>"
            f"{html.escape(name)}</td>"
        )
        for column in columns:
            background, color, text = _table_cell_display(column.variables_by_name.get(name), name in column.active_variable_names)
            style = f"padding:2px 8px;color:{color}"
            if background:
                style += f";background-color:{background}"
            rel_names = column.relation_names_by_variable.get(name, ())
            title = (
                f" title='{html.escape(chr(10).join(rel_names), quote=True)}'"
                if rel_names
                else ""
            )
            parts.append(f"<td style='{style}'{title}>{text}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _variables_text_table(source: Any, variable_names: Iterable[str] | None = None) -> str:
    """Render one source's current variables as an aligned plain-text table."""
    column = _table_column(source)
    variables = column.variables_by_name
    names = _displayed_variable_names([column], variable_names)
    rows = []
    for name in names:
        var = variables.get(name)
        value = _format_table_value(None if var is None else (var.value if var.value is not None else var.input_value))
        unit = (var.unit or "") if var is not None else ""
        rows.append((name, value, unit))
    name_w = max((len(name) for name, _, _ in rows), default=len(column.name))
    value_w = max((len(value) for _, value, _ in rows), default=0)
    lines = [column.name, "-" * (name_w + value_w + 2)]
    for name, value, unit in rows:
        line = f"{name:<{name_w}}  {value:>{value_w}}"
        lines.append(f"{line}  {unit}" if unit else line)
    return "\n".join(lines)


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

    def __post_init__(self) -> None:
        """Normalize simple user-facing fields."""
        # System produced by the most recent run(); displayed in place of the
        # loaded inputs so a solved reactor shows its reconciled values. None
        # until a mode has run. Set here so __getattr__ never intercepts it.
        self.last_system: RelationSystem | None = None
        # YAML this reactor was loaded from, if any; lets solve_reactors ship it
        # to a worker process. Set by from_yaml. Here so __getattr__ skips it.
        self.source_path: Path | None = None
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
        reactor = cls(
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
        )
        reactor.source_path = path
        return reactor

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

    def relation_system(self) -> RelationSystem:
        """Build a RelationSystem for this reactor.

        Returns:
            RelationSystem instance.
        """
        return RelationSystem(
            [var.clone() for var in self.variables.values()],
            self.relations(),
            constraints=self.constraints,
            name=self.name,
        )

    def run(self, mode: str | None = None, **options: Any) -> dict[str, Any]:
        """Build a RelationSystem, run one mode, and absorb the solved values.

        The solve runs on a clone, then each solved value replaces this reactor's
        corresponding input (both ``value`` and ``input_value`` are overwritten),
        so ``reactor.<var>`` reflects the latest solve and a re-run starts from
        it. The full solved system is kept on :attr:`last_system`, which still
        carries that run's original inputs for the input->output display.

        Args:
            mode: Optional mode override.
            **options: Mode-specific options.

        Returns:
            Result dictionary.
        """
        chosen = mode or self.mode
        system = self.relation_system()
        self.last_system = system
        if chosen == "ordered":
            result = system.ordered(order=self.relation_order or None, **options)
        else:
            result = system.run(chosen, **options)
        for name, solved in system.variables_by_name.items():
            var = self.variables.get(name)
            if var is not None and solved.value is not None:
                var.set_input(solved.value)
        return result

    def _display_source(self) -> Any:
        """Return the most recent solved system, or this reactor's inputs."""
        return self.last_system if self.last_system is not None else self

    def _repr_html_(self) -> str:
        """Rich Jupyter table of current variables (solved values after a run)."""
        return variables_table(self._display_source())

    def print_variables_table(self, *names: str) -> None:
        """Print this reactor's current variables as a plain-text table.

        Shows reconciled values after a run, otherwise the loaded inputs.

        Args:
            *names: Optional variable names to show. Defaults to all variables.
        """
        print(_variables_text_table(self._display_source(), names or None))

    def print_html_variables_table(self, *names: str) -> None:
        """Display this reactor's current variables as an HTML table (Jupyter).

        Shows reconciled values after a run (with input->output colouring,
        active-variable highlighting, and relation tooltips), otherwise the
        loaded inputs.

        Args:
            *names: Optional variable names to show. Defaults to all variables.
        """
        from IPython.display import HTML, display

        display(HTML(variables_table(self._display_source(), variable_names=names or None)))

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


def _picklable_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the picklable, display/print-relevant parts of a run result.

    A full result dict may reference relation *functions* (which do not pickle by
    reference), so worker processes return this reduced copy.
    """
    status = result.get("relation_status", {}) or {}
    clean_status = {
        name: {"enforced": bool(state.get("enforced", True)), "verified": bool(state.get("verified", False))}
        for name, state in status.items()
        if isinstance(state, Mapping)
    }
    return {
        "success": bool(result.get("success", False)),
        "termination": result.get("termination"),
        "errors": [str(error) for error in (result.get("errors") or ())],
        "warnings": [str(warning) for warning in (result.get("warnings") or ())],
        "relation_status": clean_status,
    }


def _solve_reactor_path(task: tuple[str, str, Mapping[str, Any]]) -> SolvedColumn:
    """Worker: load one reactor YAML, solve it, return a display-ready column.

    Runs in a separate process for :func:`solve_reactors`. A load/solve failure
    is returned as a failed-result column rather than raised, so one bad reactor
    does not abort the whole batch.
    """
    path, mode, options = task
    try:
        reactor = Reactor.from_yaml(path)
        reactor.run(mode, **options)
        column = _table_column(reactor.last_system)
        return column._replace(result=_picklable_result(column.result))
    except Exception as exc:  # report the failure as a column; keep the batch alive
        location = Path(path)
        return SolvedColumn(
            location.parent.name or location.stem,
            {},
            frozenset(),
            {},
            {"success": False, "termination": f"crashed: {exc!r}", "errors": [repr(exc)]},
        )


def solve_reactors(
    paths: Iterable[str | Path | Reactor],
    *,
    mode: str = "reconcile",
    workers: int | None = None,
    **options: Any,
) -> list[SolvedColumn]:
    """Solve many reactor YAMLs in parallel, one worker process each.

    Live :class:`Reactor`/:class:`RelationSystem` objects cannot cross a process
    boundary, so each worker loads its reactor from YAML and returns a picklable
    :class:`SolvedColumn`. Pass the columns straight to :func:`variables_table`.

    Args:
        paths: Reactor YAML files or directories, or reactors loaded via
            :meth:`Reactor.from_yaml` (their ``source_path`` is used).
        mode: Mode to run on each reactor.
        workers: Maximum worker processes; ``None`` lets the pool decide.
        **options: Mode options forwarded to every run.

    Returns:
        One :class:`SolvedColumn` per input, in order. Duplicate reactor names
        are disambiguated with their file path.
    """
    from concurrent.futures import ProcessPoolExecutor

    resolved: list[str] = []
    for item in paths:
        if isinstance(item, Reactor):
            if item.source_path is None:
                raise ValueError(f"Reactor {item.name!r} was not loaded from YAML; pass its path.")
            resolved.append(str(item.source_path))
        else:
            resolved.append(str(item))

    tasks = [(path, mode, options) for path in resolved]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        columns = list(executor.map(_solve_reactor_path, tasks))

    name_counts: dict[str, int] = {}
    for column in columns:
        name_counts[column.name] = name_counts.get(column.name, 0) + 1
    labelled: list[SolvedColumn] = []
    for path, column in zip(resolved, columns):
        if name_counts[column.name] > 1:
            location = Path(path)
            column = column._replace(name=f"{column.name} ({location.parent.name}/{location.name})")
        labelled.append(column)
    return labelled
