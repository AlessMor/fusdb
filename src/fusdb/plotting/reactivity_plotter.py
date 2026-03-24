"""Standalone Bokeh reactivity plotter."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from fusdb.relation_class import Relation
from fusdb.relations.reactivities import reactivity_functions as rf


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
    "BoschHale": (),  # solid
    "Hively": (4, 1, 1, 1),  # short dash, gap, dot, gap
    "CF88": (1, 3),  # dotted: dot, gap
    "NRL": (6, 2, 1, 2),  # dash, gap, dot, gap
    "ENDFB-VIII0": (1, 1),  # densely dotted: dot, short gap
    "ENDFB-VIII1": (1, 3),  # dotted: dot, gap
}

SOURCE_PRIORITY = (
    "ENDFB-VIII1",
    "ENDFB-VIII0",
    "NRL",
    "BoschHale",
    "Hively",
    "CF88",
)


@dataclass(frozen=True, slots=True)
class ReactivitySeries:
    attr_name: str
    relation: Relation
    reaction: str
    source: str
    label: str


def _source_sort_key(source: str) -> tuple[int, str]:
    if source in SOURCE_PRIORITY:
        return SOURCE_PRIORITY.index(source), source
    return len(SOURCE_PRIORITY), source


def _parse_relation_name(relation: Relation) -> tuple[str, str]:
    match = re.fullmatch(r"(?P<reaction>.+?) reactivity (?P<source>.+)", relation.name)
    if match is None:
        raise ValueError(f"Cannot parse reaction/source from relation name: {relation.name}")
    return match.group("reaction"), match.group("source")


def discover_reactivity_series() -> list[ReactivitySeries]:
    series: list[ReactivitySeries] = []
    seen_ids: set[int] = set()
    for attr_name in sorted(dir(rf)):
        if not attr_name.startswith("sigmav_"):
            continue
        relation = getattr(rf, attr_name)
        if not isinstance(relation, Relation):
            continue
        if id(relation) in seen_ids:
            continue
        seen_ids.add(id(relation))
        reaction, source = _parse_relation_name(relation)
        series.append(
            ReactivitySeries(
                attr_name=attr_name,
                relation=relation,
                reaction=reaction,
                source=source,
                label=f"{reaction} | {source}",
            )
        )
    return sorted(series, key=lambda spec: (spec.reaction, _source_sort_key(spec.source)))


def catalog_by_reaction() -> dict[str, list[ReactivitySeries]]:
    grouped: dict[str, list[ReactivitySeries]] = defaultdict(list)
    for series in discover_reactivity_series():
        grouped[series.reaction].append(series)
    return dict(grouped)


def catalog_by_source() -> dict[str, list[ReactivitySeries]]:
    grouped: dict[str, list[ReactivitySeries]] = defaultdict(list)
    for series in discover_reactivity_series():
        grouped[series.source].append(series)
    return dict(grouped)


def _validate_axis_limits(limits: tuple[float, float], *, label: str) -> tuple[float, float]:
    lower, upper = float(limits[0]), float(limits[1])
    if lower <= 0 or upper <= 0:
        raise ValueError(f"{label} must be positive for log-scaled axes.")
    if lower >= upper:
        raise ValueError(f"{label} must satisfy min < max.")
    return lower, upper


def _build_series_payload(
    *,
    x_limits: tuple[float, float],
    num_points: int,
) -> tuple[list[ReactivitySeries], np.ndarray, list[dict[str, object]]]:
    if int(num_points) < 2:
        raise ValueError("num_points must be at least 2.")
    x_limits = _validate_axis_limits(x_limits, label="x_limits")
    series = discover_reactivity_series()
    temperature_keV = np.logspace(np.log10(x_limits[0]), np.log10(x_limits[1]), int(num_points))
    payload: list[dict[str, object]] = []
    for spec in series:
        values = np.asarray(spec.relation(temperature_keV), dtype=float)
        values = np.clip(values, 1e-40, None)
        payload.append(
            {
                "x": temperature_keV.tolist(),
                "y": values.tolist(),
                "reaction": spec.reaction,
                "source": spec.source,
                "label": spec.label,
                "color": REACTION_COLORS.get(spec.reaction, "#222222"),
                "dash": SOURCE_DASHES.get(spec.source, ()),
            }
        )
    return series, temperature_keV, payload


def reactivity_plotter(
    *,
    x_limits: tuple[float, float] = (1.0, 1.0e3),
    y_limits: tuple[float, float] = (1e-30, 1e-21),
    num_points: int = 1000,
    width: int = 960,
    height: int = 620,
):
    """Return a standalone Bokeh layout for interactive reactivity exploration."""
    from bokeh.layouts import column, row
    from bokeh.models import Button, CheckboxButtonGroup, CustomJS, Div, Legend, LegendItem, TextInput
    from bokeh.plotting import figure

    x_limits = _validate_axis_limits(x_limits, label="x_limits")
    y_limits = _validate_axis_limits(y_limits, label="y_limits")
    series, _, payload = _build_series_payload(x_limits=x_limits, num_points=num_points)

    reaction_labels = sorted({spec.reaction for spec in series})
    source_labels = sorted({spec.source for spec in series}, key=_source_sort_key)

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
    plot.yaxis.axis_label = r"<σv> [m^3/s]"
    plot.grid.grid_line_alpha = 0.3

    renderers = []
    legend_items: list[LegendItem] = []
    renderer_reactions: list[str] = []
    renderer_sources: list[str] = []
    for spec, item in zip(series, payload, strict=True):
        renderer = plot.line(
            item["x"],
            item["y"],
            line_width=2.2,
            color=str(item["color"]),
            line_dash=item["dash"],
            visible=True,
        )
        renderers.append(renderer)
        legend_items.append(LegendItem(label=spec.label, renderers=[renderer], visible=True))
        renderer_reactions.append(spec.reaction)
        renderer_sources.append(spec.source)

    legend = Legend(items=legend_items, location="top_left", click_policy="hide")
    plot.add_layout(legend)

    reaction_selector = CheckboxButtonGroup(
        labels=reaction_labels,
        active=list(range(len(reaction_labels))),
        sizing_mode="stretch_width",
    )
    source_selector = CheckboxButtonGroup(
        labels=source_labels,
        active=list(range(len(source_labels))),
        sizing_mode="stretch_width",
    )

    x_min_input = TextInput(title="x min", value=str(x_limits[0]), width=120)
    x_max_input = TextInput(title="x max", value=str(x_limits[1]), width=120)
    y_min_input = TextInput(title="y min", value=str(y_limits[0]), width=120)
    y_max_input = TextInput(title="y max", value=str(y_limits[1]), width=120)
    axis_button = Button(label="Apply limits", button_type="primary", width=120)
    status = Div(text="", width=460)

    toggle_code = """
const selectedReactions = new Set(cbReactions.active.map((index) => reactionLabels[index]));
const selectedSources = new Set(cbSources.active.map((index) => sourceLabels[index]));
for (let i = 0; i < renderers.length; i++) {
  const isVisible = selectedReactions.has(rendererReactions[i]) && selectedSources.has(rendererSources[i]);
  renderers[i].visible = isVisible;
  legendItems[i].visible = isVisible;
}
status.text = "";
"""
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
        code=toggle_code,
    )
    reaction_selector.js_on_change("active", toggle_callback)
    source_selector.js_on_change("active", toggle_callback)

    axis_code = """
const parseLimit = (value) => Number(value);
const xMin = parseLimit(xMinInput.value);
const xMax = parseLimit(xMaxInput.value);
const yMin = parseLimit(yMinInput.value);
const yMax = parseLimit(yMaxInput.value);

if (!(xMin > 0 && xMax > 0 && yMin > 0 && yMax > 0)) {
  status.text = "<span style='color:#b00020'>Axis limits must be positive.</span>";
  return;
}
if (!(xMin < xMax && yMin < yMax)) {
  status.text = "<span style='color:#b00020'>Each axis must satisfy min &lt; max.</span>";
  return;
}

plot.x_range.start = xMin;
plot.x_range.end = xMax;
plot.y_range.start = yMin;
plot.y_range.end = yMax;
status.text = "";
"""
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
            code=axis_code,
        )
    )

    label_style = "font-weight:600; min-width:90px; padding-top:6px;"
    reactions_row = row(
        Div(text=f"<div style='{label_style}'>Reactions</div>", width=100),
        reaction_selector,
        sizing_mode="stretch_width",
    )
    sources_row = row(
        Div(text=f"<div style='{label_style}'>Sources</div>", width=100),
        source_selector,
        sizing_mode="stretch_width",
    )
    limits_row = row(
        Div(text=f"<div style='{label_style}'>Limits</div>", width=100),
        x_min_input,
        x_max_input,
        y_min_input,
        y_max_input,
        axis_button,
        status,
        sizing_mode="stretch_width",
    )

    return column(
        plot,
        reactions_row,
        sources_row,
        limits_row,
        sizing_mode="stretch_width",
    )


def save_reactivity_plotter_html(
    path: str | Path,
    *,
    x_limits: tuple[float, float] = (1.0, 1.0e3),
    y_limits: tuple[float, float] = (1e-30, 1e-21),
    num_points: int = 1000,
    width: int = 960,
    height: int = 620,
) -> Path:
    """Write the standalone reactivity plotter HTML file and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_reactivity_plotter_html(
            x_limits=x_limits,
            y_limits=y_limits,
            num_points=num_points,
            width=width,
            height=height,
        ),
        encoding="utf-8",
    )
    return output_path


def render_reactivity_plotter_html(
    *,
    x_limits: tuple[float, float] = (1.0, 1.0e3),
    y_limits: tuple[float, float] = (1e-30, 1e-21),
    num_points: int = 1000,
    width: int = 960,
    height: int = 620,
) -> str:
    """Return the standalone reactivity plotter HTML document."""
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(
        reactivity_plotter(
            x_limits=x_limits,
            y_limits=y_limits,
            num_points=num_points,
            width=width,
            height=height,
        ),
        CDN,
        "Fusion Reactivity Plotter",
    )
