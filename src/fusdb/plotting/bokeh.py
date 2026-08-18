"""Shared scaffolding for the interactive Bokeh explorer apps.

Hosts the pieces that :mod:`fusdb.plotting.reactivity` and
:mod:`fusdb.plotting.atomic_physics` would otherwise copy from each other: the
axis-limit validation and log sample grid, the log-log figure scaffold, the
two-dimension visibility filter, the axis-limit controls, labelled widget rows
and the standalone-HTML embedding. ``bokeh`` is imported lazily inside each
helper, so importing this module (or ``fusdb.plotting``) does not pull it in.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .data import CurveSet

_LABEL_STYLE = "font-weight:600; min-width:90px; padding-top:6px;"


def validate_axis_limits(limits: tuple[float, float], *, label: str) -> tuple[float, float]:
    lower, upper = float(limits[0]), float(limits[1])
    if lower <= 0 or upper <= 0:
        raise ValueError(f"{label} must be positive for log-scaled axes.")
    if lower >= upper:
        raise ValueError(f"{label} must satisfy min < max.")
    return lower, upper


def log_grid(x_limits: tuple[float, float], num_points: int) -> np.ndarray:
    """Return a log-spaced sample grid across positive ``x_limits``."""
    if int(num_points) < 2:
        raise ValueError("num_points must be at least 2.")
    return np.logspace(np.log10(x_limits[0]), np.log10(x_limits[1]), int(num_points))


def log_log_figure(
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    width: int,
    height: int,
    title: str,
    x_label: str,
    y_label: str,
) -> Any:
    """Return the log-log explorer figure both apps are built on."""
    from bokeh.plotting import figure

    plot = figure(
        width=width,
        height=height,
        x_axis_type="log",
        y_axis_type="log",
        x_range=x_limits,
        y_range=y_limits,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        title=title,
        sizing_mode="stretch_width",
    )
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.grid.grid_line_alpha = 0.3
    return plot


def move_legends_below(
    plot: Any,
    legends: Sequence[Any] | None = None,
    *,
    ncols: int | None = None,
) -> None:
    """Place the plot's legends in a compact grid below the figure.

    Legends created through Bokeh's ``legend_label``/``legend_group`` glyph
    arguments start in the figure's overlay area. Remove them from their
    current layout slot before adding them below so the same legend model is
    not present in two layout slots at once.
    """
    if ncols is not None and ncols < 1:
        raise ValueError("ncols must be at least 1.")
    if legends is None:
        legends = list(plot.legend)

    if legends and getattr(plot, "frame_height", None) is None:
        plot.frame_height = plot.height
        plot.height = None

    for legend in legends:
        legend.orientation = "horizontal"
        legend.margin = 5
        legend.padding = 5
        item_widths = _legend_item_widths(legend)
        legend.ncols = min(
            ncols
            if ncols is not None
            else _columns_for_width(
                item_widths,
                max(
                    1,
                    int(getattr(plot, "width", 960))
                    - 2 * (legend.margin + legend.padding),
                ),
                legend.spacing,
            ),
            max(len(legend.items), 1),
        )
        for location in ("above", "below", "left", "right", "center"):
            layout = getattr(plot, location)
            if legend in layout:
                layout.remove(legend)
        plot.add_layout(legend, "below")
        if ncols is None:
            _link_responsive_legend(plot, legend, item_widths)


def _legend_item_widths(legend: Any) -> list[int]:
    """Estimate each legend item's rendered width in pixels."""
    font_size = str(legend.label_text_font_size)
    character_width = 6 if font_size.endswith("pt") and float(font_size[:-2]) <= 8 else 7
    item_widths = []
    for item in legend.items:
        label = item.label
        text = getattr(label, "value", None) or getattr(label, "field", None) or str(label)
        label_width = max(50, min(len(str(text)) * character_width, 280))
        item_widths.append(legend.glyph_width + legend.label_standoff + label_width)
    return item_widths


def _columns_for_width(item_widths: Sequence[int], available_width: int, spacing: int) -> int:
    """Return the largest row-major column count fitting ``available_width``."""
    if not item_widths:
        return 1
    best = 1
    for columns in range(1, len(item_widths) + 1):
        column_widths = [
            max(item_widths[column::columns])
            for column in range(columns)
            if item_widths[column::columns]
        ]
        required_width = sum(column_widths) + spacing * max(columns - 1, 0)
        if required_width <= available_width:
            best = columns
    return best


def _link_responsive_legend(plot: Any, legend: Any, item_widths: Sequence[int]) -> None:
    """Reflow a legend whenever Bokeh changes the rendered plot width."""
    from bokeh.models import CustomJS

    callback = CustomJS(
        args=dict(plot=plot, legend=legend, itemWidths=list(item_widths)),
        code="""
const availableWidth = Math.max(
  1,
  (plot.inner_width ?? plot.width ?? 1) - 2 * (legend.margin + legend.padding),
);
let best = 1;
for (let columns = 1; columns <= itemWidths.length; columns++) {
  const columnWidths = new Array(columns).fill(0);
  for (let index = 0; index < itemWidths.length; index++) {
    const column = index % columns;
    columnWidths[column] = Math.max(columnWidths[column], itemWidths[index]);
  }
  const requiredWidth =
    columnWidths.reduce((total, width) => total + width, 0) +
    legend.spacing * Math.max(columns - 1, 0);
  if (requiredWidth <= availableWidth) {
    best = columns;
  }
}
if (legend.ncols !== best) {
  legend.ncols = best;
}
""",
    )
    plot.js_on_change("inner_width", callback)


def link_two_filter_visibility(
    selector_a: Any,
    selector_b: Any,
    button_values_a: Sequence[str],
    button_values_b: Sequence[str],
    renderer_values_a: Sequence[str],
    renderer_values_b: Sequence[str],
    renderers: Sequence[Any],
    legend_items: Sequence[Any],
    status: Any,
) -> None:
    """Wire two checkbox groups to renderer/legend visibility (AND of both).

    ``button_values_*`` map a selector's button index to its filter value (the
    display label on the button may differ); ``renderer_values_*`` give the
    filter value of each renderer/legend item.
    """
    from bokeh.models import CustomJS

    callback = CustomJS(
        args=dict(
            selectorA=selector_a,
            selectorB=selector_b,
            buttonValuesA=list(button_values_a),
            buttonValuesB=list(button_values_b),
            rendererValuesA=list(renderer_values_a),
            rendererValuesB=list(renderer_values_b),
            renderers=list(renderers),
            legendItems=list(legend_items),
            status=status,
        ),
        code="""
const selectedA = new Set(selectorA.active.map((index) => buttonValuesA[index]));
const selectedB = new Set(selectorB.active.map((index) => buttonValuesB[index]));
for (let i = 0; i < renderers.length; i++) {
  const isVisible = selectedA.has(rendererValuesA[i]) && selectedB.has(rendererValuesB[i]);
  renderers[i].visible = isVisible;
  legendItems[i].visible = isVisible;
}
status.text = "";
""",
    )
    selector_a.js_on_change("active", callback)
    selector_b.js_on_change("active", callback)


def axis_limit_controls(plot: Any, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> tuple[list, Any]:
    """Return ``(widgets, status)``: min/max inputs + apply button wired to the plot ranges."""
    from bokeh.models import Button, CustomJS, Div, TextInput

    x_min_input = TextInput(title="x min", value=str(x_limits[0]), width=120)
    x_max_input = TextInput(title="x max", value=str(x_limits[1]), width=120)
    y_min_input = TextInput(title="y min", value=str(y_limits[0]), width=120)
    y_max_input = TextInput(title="y max", value=str(y_limits[1]), width=120)
    axis_button = Button(label="Apply limits", button_type="primary", width=120)
    status = Div(text="", width=460)

    axis_button.js_on_click(
        CustomJS(
            args=dict(
                plot=plot,
                xMinInput=x_min_input,
                xMaxInput=x_max_input,
                yMinInput=y_min_input,
                yMaxInput=y_max_input,
                status=status,
            ),
            code="""
const xMin = Number(xMinInput.value), xMax = Number(xMaxInput.value);
const yMin = Number(yMinInput.value), yMax = Number(yMaxInput.value);
if (!(xMin > 0 && xMax > 0 && yMin > 0 && yMax > 0)) {
  status.text = "<span style='color:#b00020'>Axis limits must be positive.</span>";
  return;
}
if (!(xMin < xMax && yMin < yMax)) {
  status.text = "<span style='color:#b00020'>Each axis must satisfy min &lt; max.</span>";
  return;
}
plot.x_range.start = xMin; plot.x_range.end = xMax;
plot.y_range.start = yMin; plot.y_range.end = yMax;
status.text = "";
""",
        )
    )
    return [x_min_input, x_max_input, y_min_input, y_max_input, axis_button, status], status


def explorer_layout(
    plot: Any,
    *,
    legend_items: Sequence[Any],
    option_controls: Sequence[tuple[str, Any]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    legend_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Build the standard explorer: plot + legend, options, then limits.

    Returns ``(layout, status)`` so the caller can connect its option widgets
    to the shared status message used by the axis-limit controls.
    """
    from bokeh.layouts import column, row
    from bokeh.models import Div, Legend

    legend = Legend(
        items=list(legend_items),
        **{"click_policy": "hide", **(legend_kwargs or {})},
    )
    move_legends_below(plot, [legend])
    limit_widgets, status = axis_limit_controls(plot, x_limits, y_limits)
    rows = [
        row(
            Div(text=f"<div style='{_LABEL_STYLE}'>{label}</div>", width=100),
            control,
            sizing_mode="stretch_width",
        )
        for label, control in option_controls
    ]
    limit_row = row(
        Div(text=f"<div style='{_LABEL_STYLE}'>Limits</div>", width=100),
        *limit_widgets,
        sizing_mode="stretch_width",
    )
    return (
        column(
            plot,
            *rows,
            limit_row,
            sizing_mode="stretch_width",
        ),
        status,
    )


def bokeh_curve_set(
    data: CurveSet,
    *,
    plot: Any | None = None,
    width: int = 960,
    height: int = 620,
    tools: str = "pan,wheel_zoom,box_zoom,reset,save",
    legend: bool = True,
) -> tuple[Any, list[Any], list[Any]]:
    """Render ``data`` with Bokeh and return ``(figure, renderers, sources)``.

    Passing an existing ``plot`` lets domain explorers add filters and widgets
    without duplicating the conversion from :class:`Curve` to Bokeh sources.
    """
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure

    if plot is None:
        plot = figure(
            width=width,
            height=height,
            x_axis_type=data.xscale,
            y_axis_type=data.yscale,
            title=data.title or "",
            tools=tools,
            active_scroll="wheel_zoom",
            sizing_mode="stretch_width",
        )
    renderers, sources = [], []
    for curve in data.curves:
        source = ColumnDataSource(curve.source_data())
        style = dict(curve.style)
        marker_only = style.pop("marker_only", False)
        line_width = style.pop("linewidth", style.pop("line_width", 2))
        legend_kw = {"legend_label": curve.label} if legend else {}
        if marker_only:
            marker_size = style.pop("markersize", style.pop("size", 8))
            renderer = plot.scatter("x", "y", source=source, size=marker_size, **legend_kw, **style)
        else:
            renderer = plot.line("x", "y", source=source, line_width=line_width, **legend_kw, **style)
        renderers.append(renderer)
        sources.append(source)
    plot.xaxis.axis_label = data.xlabel or ""
    plot.yaxis.axis_label = data.ylabel or ""
    plot.grid.grid_line_alpha = 0.3
    if legend and plot.legend:
        move_legends_below(plot)
        plot.legend.click_policy = "hide"
    return plot, renderers, sources
