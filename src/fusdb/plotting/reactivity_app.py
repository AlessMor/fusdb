"""Interactive Bokeh reactivity plotter.

A standalone, client-side interactive ``<sigma v>(T_i)`` explorer: every
reaction/parametrisation discovered in the relation registry is drawn as a
toggleable curve, with reaction and source filters, a hide-on-legend-click, and
adjustable log-axis limits. Used for the embedded docs widget
(``code_docs/reactivity_plotter.html``) and reusable on its own.

This module imports ``bokeh`` lazily inside the builders, so it is *not* pulled
in by ``import fusdb.plotting``; install the ``plotting`` (or ``docs``) extra to
use it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from fusdb.registry import RELATIONS

# Stable colour per reaction and dash pattern per parametrisation, so the same
# reaction reads consistently across its sources.
REACTION_COLORS = {
    "DT": "#1f77b4",
    "DD": "#556b2f",
    "DDn": "#ff7f0e",
    "DDp": "#b8860b",
    "DHe3": "#2ca02c",
    "TT": "#d62728",
    "He3He3": "#9467bd",
    "THe3": "#8c564b",
    "THe3_D": "#a0522d",
    "THe3_np": "#d2691e",
}
SOURCE_DASHES = {
    "BoschHale": (),
    "Hively": (4, 1, 1, 1),
    "CF88": (1, 3),
    "NRL": (6, 2, 1, 2),
    "ENDFB-VIII0": (1, 1),
    "ENDFB-VIII1": (1, 3),
}
SOURCE_PRIORITY = ("ENDFB-VIII1", "ENDFB-VIII0", "NRL", "BoschHale", "Hively", "CF88")

_NAME_RE = re.compile(r"(?P<reaction>.+?) reactivity (?P<source>.+)")


def _source_sort_key(source: str) -> tuple[int, str]:
    if source in SOURCE_PRIORITY:
        return SOURCE_PRIORITY.index(source), source
    return len(SOURCE_PRIORITY), source


def discover_reactivity_series() -> list[tuple[str, str, str, object]]:
    """Return ``(reaction, source, label, relation)`` for every reactivity curve.

    Discovered from the registry: relations mapping ``T_i`` to a single
    ``sigmav_*`` output whose name parses as ``"<reaction> reactivity <source>"``.
    Sorted by reaction, then by source preference.
    """
    series: list[tuple[str, str, str, object]] = []
    for relation in RELATIONS:
        outputs = relation.outputs
        if relation.input_names != ("T_i",) or len(outputs) != 1 or not outputs[0].startswith("sigmav_"):
            continue
        match = _NAME_RE.fullmatch(relation.name)
        if match is None:
            continue
        reaction, source = match.group("reaction"), match.group("source")
        series.append((reaction, source, f"{reaction} | {source}", relation))
    return sorted(series, key=lambda item: (item[0], _source_sort_key(item[1])))


def _validate_axis_limits(limits: tuple[float, float], *, label: str) -> tuple[float, float]:
    lower, upper = float(limits[0]), float(limits[1])
    if lower <= 0 or upper <= 0:
        raise ValueError(f"{label} must be positive for log-scaled axes.")
    if lower >= upper:
        raise ValueError(f"{label} must satisfy min < max.")
    return lower, upper


def reactivity_app(
    *,
    x_limits: tuple[float, float] = (1.0, 1.0e3),
    y_limits: tuple[float, float] = (1e-30, 1e-21),
    num_points: int = 1000,
    width: int = 960,
    height: int = 620,
):
    """Return a standalone Bokeh layout for interactive reactivity exploration.

    Args:
        x_limits: Initial ion-temperature axis range in keV (log scale).
        y_limits: Initial reactivity axis range in m^3/s (log scale).
        num_points: Samples per curve across ``x_limits``.
        width, height: Plot size in pixels.

    Returns:
        A Bokeh layout model (embed with :func:`render_reactivity_app_html`).
    """
    from bokeh.layouts import column, row
    from bokeh.models import Button, CheckboxButtonGroup, CustomJS, Div, Legend, LegendItem, TextInput
    from bokeh.plotting import figure

    x_limits = _validate_axis_limits(x_limits, label="x_limits")
    y_limits = _validate_axis_limits(y_limits, label="y_limits")
    if int(num_points) < 2:
        raise ValueError("num_points must be at least 2.")

    series = discover_reactivity_series()
    temperature_keV = np.logspace(np.log10(x_limits[0]), np.log10(x_limits[1]), int(num_points))

    plot = figure(
        width=width,
        height=height,
        x_axis_type="log",
        y_axis_type="log",
        x_range=x_limits,
        y_range=y_limits,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        title="Fusion Reactivities",
        sizing_mode="stretch_width",
    )
    plot.xaxis.axis_label = "Ion temperature [keV]"
    plot.yaxis.axis_label = "⟨σv⟩ [m^3/s]"
    plot.grid.grid_line_alpha = 0.3

    renderers = []
    legend_items: list = []
    renderer_reactions: list[str] = []
    renderer_sources: list[str] = []
    for reaction, source, label, relation in series:
        # ``evaluate`` returns raw function values (no domain enforcement), so
        # edge NaNs/zeros are tolerated here and clipped for the log axis. Some
        # parametrisations divide/power through invalid values at the edges; the
        # resulting numpy warnings are expected and silenced.
        with np.errstate(all="ignore"):
            raw = np.asarray(relation.evaluate({"T_i": temperature_keV}), dtype=float)
        values = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 1e-40, None)
        renderer = plot.line(
            temperature_keV.tolist(),
            values.tolist(),
            line_width=2.2,
            color=REACTION_COLORS.get(reaction, "#222222"),
            line_dash=SOURCE_DASHES.get(source, ()),
            visible=True,
        )
        renderers.append(renderer)
        legend_items.append(LegendItem(label=label, renderers=[renderer], visible=True))
        renderer_reactions.append(reaction)
        renderer_sources.append(source)

    plot.add_layout(Legend(items=legend_items, location="top_left", click_policy="hide"))

    reaction_labels = sorted({reaction for reaction, *_ in series})
    source_labels = sorted({source for _, source, *_ in series}, key=_source_sort_key)
    reaction_selector = CheckboxButtonGroup(
        labels=reaction_labels, active=list(range(len(reaction_labels))), sizing_mode="stretch_width"
    )
    source_selector = CheckboxButtonGroup(
        labels=source_labels, active=list(range(len(source_labels))), sizing_mode="stretch_width"
    )

    x_min_input = TextInput(title="x min", value=str(x_limits[0]), width=120)
    x_max_input = TextInput(title="x max", value=str(x_limits[1]), width=120)
    y_min_input = TextInput(title="y min", value=str(y_limits[0]), width=120)
    y_max_input = TextInput(title="y max", value=str(y_limits[1]), width=120)
    axis_button = Button(label="Apply limits", button_type="primary", width=120)
    status = Div(text="", width=460)

    toggle_callback = CustomJS(
        args=dict(
            cbReactions=reaction_selector,
            cbSources=source_selector,
            renderers=renderers,
            reactionLabels=reaction_labels,
            sourceLabels=source_labels,
            rendererReactions=renderer_reactions,
            rendererSources=renderer_sources,
            legendItems=legend_items,
            status=status,
        ),
        code="""
const selectedReactions = new Set(cbReactions.active.map((index) => reactionLabels[index]));
const selectedSources = new Set(cbSources.active.map((index) => sourceLabels[index]));
for (let i = 0; i < renderers.length; i++) {
  const isVisible = selectedReactions.has(rendererReactions[i]) && selectedSources.has(rendererSources[i]);
  renderers[i].visible = isVisible;
  legendItems[i].visible = isVisible;
}
status.text = "";
""",
    )
    reaction_selector.js_on_change("active", toggle_callback)
    source_selector.js_on_change("active", toggle_callback)

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

    label_style = "font-weight:600; min-width:90px; padding-top:6px;"
    return column(
        plot,
        row(Div(text=f"<div style='{label_style}'>Reactions</div>", width=100), reaction_selector,
            sizing_mode="stretch_width"),
        row(Div(text=f"<div style='{label_style}'>Sources</div>", width=100), source_selector,
            sizing_mode="stretch_width"),
        row(Div(text=f"<div style='{label_style}'>Limits</div>", width=100),
            x_min_input, x_max_input, y_min_input, y_max_input, axis_button, status,
            sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


def render_reactivity_app_html(
    *,
    x_limits: tuple[float, float] = (1.0, 1.0e3),
    y_limits: tuple[float, float] = (1e-30, 1e-21),
    num_points: int = 1000,
    width: int = 960,
    height: int = 620,
    title: str = "Fusion Reactivity Plotter",
) -> str:
    """Return a self-contained interactive HTML document (BokehJS from CDN)."""
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    app = reactivity_app(
        x_limits=x_limits, y_limits=y_limits, num_points=num_points, width=width, height=height
    )
    return file_html(app, CDN, title)


def save_reactivity_app_html(path: str | Path, **kwargs: object) -> Path:
    """Write the interactive reactivity plotter HTML to ``path`` and return it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_reactivity_app_html(**kwargs), encoding="utf-8")  # type: ignore[arg-type]
    return output_path
