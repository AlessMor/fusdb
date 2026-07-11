"""Variable-table rendering for reactors, relation systems and solved columns.

Presentation only: everything here reads reactors/systems through a small
duck-typed surface (see :func:`_table_column`) and produces HTML or plain-text
tables.  Kept out of :mod:`fusdb.reactor` so the domain modules do not depend
on rendering code (``RelationSystem._repr_html_`` imports from here).
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np

from ..registry import VARIABLES


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
    # Reactor: its Variable records supply inputs/current values and tolerances.
    records = source.variables.values()
    return SolvedColumn(
        source.name,
        {v.name: v.input_value for v in records if v.input_value is not None},
        {v.name: v.value for v in records if v.value is not None},
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


def variables_table(*sources: Any, variable_names: Iterable[str] | None = None) -> str:
    """Render current variable values for one or more reactors/systems as HTML.

    Each positional source is a :class:`Reactor`, a :class:`RelationSystem`, or a
    :class:`SolvedColumn` (e.g. from :func:`fusdb.reactor.solve_reactors`);
    columns are sources and rows are variables. Reactor columns show current
    values; solved systems and columns additionally highlight active variables,
    colour input->output changes, add relation tooltips, and colour the header
    by solve success. ``variable_names`` overrides the row order/subset; when
    omitted, all active and user-supplied variables are shown.

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
            background, color, text = _table_cell_display(
                column.inputs.get(name), column.values.get(name),
                column.rel_tols.get(name, 0.0), column.abs_tols.get(name, 0.0),
                name in column.active_variable_names,
            )
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
    names = _displayed_variable_names([column], variable_names)
    rows = []
    for name in names:
        current = column.values.get(name)
        value = _format_table_value(current if current is not None else column.inputs.get(name))
        unit = VARIABLES.get(name).unit if name in VARIABLES else ""
        rows.append((name, value, unit))
    name_w = max((len(name) for name, _, _ in rows), default=len(column.name))
    value_w = max((len(value) for _, value, _ in rows), default=0)
    lines = [column.name, "-" * (name_w + value_w + 2)]
    for name, value, unit in rows:
        line = f"{name:<{name_w}}  {value:>{value_w}}"
        lines.append(f"{line}  {unit}" if unit else line)
    return "\n".join(lines)
