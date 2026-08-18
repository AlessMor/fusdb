"""Prepare and render variable tables for reactors, systems, and solve results."""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ..registry import VARIABLES


@dataclass(frozen=True)
class TableCell:
    """A display-ready table cell, independent of HTML or text rendering."""

    text: str
    foreground: str = "#000000"
    background: str = ""
    tooltip: str = ""


@dataclass(frozen=True)
class TableData:
    """A table with already-formatted cells for HTML and plain-text renderers."""

    headers: Sequence[str]
    rows: Sequence[tuple[str, Sequence[TableCell]]]
    header_colors: Sequence[str] = ()

    def __post_init__(self) -> None:
        headers = tuple(self.headers)
        rows = tuple((str(name), tuple(cells)) for name, cells in self.rows)
        if any(len(cells) != len(headers) for _, cells in rows):
            raise ValueError("Every TableData row must contain one cell per header.")
        colors = tuple(self.header_colors) or tuple("#000000" for _ in headers)
        if len(colors) != len(headers):
            raise ValueError("TableData header_colors must match headers.")
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "header_colors", colors)


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


def _table_cell_display(input_value: Any, value: Any, rel_tol: float, abs_tol: float, used: bool) -> tuple[str, str, str]:
    """Return ``(background, foreground, html_text)`` for one variable cell."""
    background = ""
    color = "#000000"
    text = ""
    has_input = input_value is not None
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
                float(abs_tol or 0.0),
                float(rel_tol or 0.0) * scale,
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
    parallel solve (see :func:`fusdb.reactor.solve_reactors`).
    """

    name: str
    inputs: Mapping[str, Any]
    values: Mapping[str, Any]
    rel_tols: Mapping[str, float]
    abs_tols: Mapping[str, float]
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
    if hasattr(source, "variable_roles"):  # RelationSystem
        relations: dict[str, list[str]] = {}
        for rel in getattr(source, "relations", ()):
            for variable_name in rel.variables:
                relations.setdefault(variable_name, []).append(rel.name)
        return SolvedColumn(
            source.name,
            dict(source.inputs),
            dict(source.values),
            dict(source.rel_tols),
            dict(source.abs_tols),
            frozenset(source.active_variable_names),
            {name: tuple(dict.fromkeys(names)) for name, names in relations.items()},
            getattr(source, "last_result", None) or {},
        )
    # Reactor: its Variable records supply the declared inputs and tolerances;
    # current values read through last_plan's solved state when present (a
    # Variable's own declaration never changes after a solve), falling back
    # to the declaration itself for an unsolved reactor.
    records = source.variables.values()
    last_plan = getattr(source, "last_plan", None)
    solved = last_plan.values if last_plan is not None else {}
    current = {v.name: solved.get(v.name, v.value) for v in records}
    return SolvedColumn(
        source.name,
        {v.name: v.input_value for v in records if v.input_value is not None},
        {name: value for name, value in current.items() if value is not None},
        {v.name: float(v.rel_tol or 0.0) for v in records},
        {v.name: float(v.abs_tol or 0.0) for v in records},
        frozenset(),
        {},
        {},
    )


def _displayed_variable_names(columns: Iterable[SolvedColumn], variable_names: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve the row order/subset: the explicit list, or active + supplied."""
    if variable_names is not None:
        return tuple(variable_names)
    names: set[str] = set()
    for column in columns:
        names.update(column.active_variable_names)
        names.update(column.inputs)
    return _sort_table_variable_names(names)


def variable_table_data(*sources: Any, variable_names: Iterable[str] | None = None) -> TableData:
    """Prepare current variable values for HTML or plain-text presentation.

    Each positional source is a :class:`Reactor`, a :class:`RelationSystem`, or a
    :class:`SolvedColumn` (e.g. from :func:`fusdb.reactor.solve_reactors`);
    columns are sources and rows are variables. Reactor columns show current
    values; solved systems and columns additionally highlight active variables,
    colour input->output changes, add relation tooltips, and colour the header
    by solve success. ``variable_names`` overrides the row order/subset; when
    omitted, all active and user-supplied variables are shown.

    The returned data contains the same solve-status and input/output change
    information without committing to a renderer.
    """
    columns = [_table_column(source) for source in sources]
    ordered_names = _displayed_variable_names(columns, variable_names)
    rows = []
    for name in ordered_names:
        cells = []
        for column in columns:
            background, color, text = _table_cell_display(
                column.inputs.get(name), column.values.get(name),
                column.rel_tols.get(name, 0.0), column.abs_tols.get(name, 0.0),
                name in column.active_variable_names,
            )
            rel_names = column.relation_names_by_variable.get(name, ())
            cells.append(TableCell(text, foreground=color, background=background, tooltip="\n".join(rel_names)))
        rows.append((name, cells))
    header_colors = tuple(
        "#1EFF00" if column.result.get("success") else "#c00000" if column.result else "#000000"
        for column in columns
    )
    return TableData([column.name for column in columns], rows, header_colors)


def render_table(data: TableData, *, format: str = "html", title: str | None = None) -> str:
    """Render prepared table data as HTML or aligned plain text."""
    if format == "html":
        parts = ["<table style='border-collapse:collapse;font-size:0.8em'>"]
        parts.append("<tr><th style='text-align:left;padding:2px 8px'>variable</th>")
        for header, color in zip(data.headers, data.header_colors, strict=True):
            parts.append(f"<th style='padding:2px 8px;color:{color}'>{html.escape(header)}</th>")
        parts.append("</tr>")
        for name, cells in data.rows:
            parts.append(f"<tr><td style='text-align:left;padding:2px 8px;font-weight:bold'>{html.escape(name)}</td>")
            for cell in cells:
                style = f"padding:2px 8px;color:{cell.foreground}"
                if cell.background:
                    style += f";background-color:{cell.background}"
                tooltip = f" title='{html.escape(cell.tooltip, quote=True)}'" if cell.tooltip else ""
                parts.append(f"<td style='{style}'{tooltip}>{cell.text}</td>")
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)
    if format != "text":
        raise ValueError("Table format must be 'html' or 'text'.")
    if len(data.headers) != 1:
        raise ValueError("Plain-text rendering requires exactly one table column.")
    rows = [(name, cell.text.replace("<b>", "").replace("</b>", "").replace("&rarr;", "->")) for name, (cell,) in data.rows]
    name_w = max((len(name) for name, _ in rows), default=len(title or data.headers[0]))
    value_w = max((len(value) for _, value in rows), default=0)
    lines = [title or data.headers[0], "-" * (name_w + value_w + 2)]
    lines.extend(f"{name:<{name_w}}  {value:>{value_w}}" for name, value in rows)
    return "\n".join(lines)
