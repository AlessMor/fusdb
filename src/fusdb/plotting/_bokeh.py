"""Shared scaffolding for the interactive Bokeh explorer apps.

Hosts the pieces that :mod:`fusdb.plotting.reactivity` and
:mod:`fusdb.plotting.atomic_physics` would otherwise copy from each other: the
axis-limit validation and log sample grid, the log-log figure scaffold, the
two-dimension visibility filter, the axis-limit controls, labelled widget rows
and the standalone-HTML embedding. ``bokeh`` is imported lazily inside each
helper, so importing this module (or ``fusdb.plotting``) does not pull it in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

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


def labeled_row(label: str, *widgets: Any) -> Any:
    """Return a stretch-width row with a fixed-width bold label in front."""
    from bokeh.layouts import row
    from bokeh.models import Div

    return row(
        Div(text=f"<div style='{_LABEL_STYLE}'>{label}</div>", width=100),
        *widgets,
        sizing_mode="stretch_width",
    )


def model_html(model: Any, title: str) -> str:
    """Return a self-contained interactive HTML document (BokehJS from CDN)."""
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(model, CDN, title)


def write_html(path: str | Path, html: str) -> Path:
    """Write ``html`` to ``path`` (creating parent directories) and return it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
