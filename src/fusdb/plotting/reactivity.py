"""Fusion reactivity curves, ``<sigma v>`` versus ion temperature.

A single source of truth for the reactivity plotters. Reaction sources are
discovered from the relation registry, so new reactions and parametrisations
appear without editing this module. Two representations share that discovery:

* :func:`reactivity_curves` -- backend-neutral log-log curve data, one preferred
  curve per reaction.
* :func:`reactivity_app` -- a standalone, client-side interactive Bokeh explorer
  with every parametrisation drawn as a toggleable curve (used for the embedded
  docs widget ``code_docs/reactivity_plotter.html``).

``bokeh`` is imported lazily inside the interactive builders, so it is *not*
pulled in by ``import fusdb.plotting``; install the ``plotting`` (or ``docs``)
extra to use the interactive app.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

import numpy as np

from fusdb.registry import RELATIONS

from .bokeh import (
    explorer_layout,
    link_two_filter_visibility,
    log_grid,
    log_log_figure,
    validate_axis_limits,
)
from .data import Curve, CurveSet
from .bokeh import bokeh_curve_set

# Preferred parametrisation per reaction, best-first. The single ordering used by
# both the static plot (which keeps one curve per reaction) and the interactive
# app (which orders its source filter and legend by it).
SOURCE_PREFERENCE = ("BoschHale", "CF88", "Hively", "NRL", "ENDFB-VIII1", "ENDFB-VIII0")
DEFAULT_TEMPERATURE_KEV = np.logspace(0.0, 2.7, 240)

# Stable colour per reaction and dash pattern per parametrisation (interactive
# app), so the same reaction reads consistently across its sources.
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

# Relation names follow the ``"<reaction> reactivity <source>"`` convention.
_NAME_RE = re.compile(r"(?P<reaction>.+?) reactivity (?P<source>.+)")

# A reactivity source is either a ``@relation`` object or a plain ``f(T_i=...)``.
ReactivitySource = Callable[..., Any]


def _is_reactivity_relation(relation: Any) -> bool:
    """Return whether a relation maps ``T_i`` to a single ``sigmav_*`` output."""
    outputs = relation.outputs
    return relation.input_names == ("T_i",) and len(outputs) == 1 and outputs[0].startswith("sigmav_")


def _source_rank(source: str) -> int:
    """Return the preference index of a parametrisation (lower is better)."""
    return SOURCE_PREFERENCE.index(source) if source in SOURCE_PREFERENCE else len(SOURCE_PREFERENCE)


def _evaluate(source: ReactivitySource, temperature_keV: np.ndarray) -> np.ndarray:
    """Evaluate a reactivity source on a temperature grid as a float array."""
    if hasattr(source, "evaluate"):  # Relation object
        values = source.evaluate({"T_i": temperature_keV})
    else:  # plain callable f(T_i=...)
        values = source(T_i=temperature_keV)
    return np.asarray(values, dtype=float)


def discover_reactivity_series() -> list[tuple[str, str, str, Any]]:
    """Return ``(reaction, source, label, relation)`` for every reactivity curve.

    Discovered from the registry: relations mapping ``T_i`` to a single
    ``sigmav_*`` output whose name parses as ``"<reaction> reactivity <source>"``.
    Sorted by reaction, then by :data:`SOURCE_PREFERENCE`.
    """
    series: list[tuple[str, str, str, Any]] = []
    for relation in RELATIONS:
        if not _is_reactivity_relation(relation):
            continue
        match = _NAME_RE.fullmatch(relation.name)
        if match is None:
            continue
        reaction, source = match.group("reaction"), match.group("source")
        series.append((reaction, source, f"{reaction} | {source}", relation))
    return sorted(series, key=lambda item: (item[0], _source_rank(item[1])))


def _default_reactivities() -> dict[str, ReactivitySource]:
    """Return one preferred reactivity relation per reaction, keyed by label.

    For each reaction the source earliest in :data:`SOURCE_PREFERENCE` is kept.

    Returns:
        Mapping of ``"<reaction> (<source>)"`` to the chosen relation.
    """
    chosen: dict[str, ReactivitySource] = {}
    seen: set[str] = set()
    # Series is preference-sorted within each reaction, so the first source seen
    # for a reaction is the preferred one.
    for reaction, source, _label, relation in discover_reactivity_series():
        if reaction in seen:
            continue
        seen.add(reaction)
        chosen[f"{reaction} ({source})"] = relation
    return chosen


def reactivity_curves(
    reactions: Mapping[str, ReactivitySource] | None = None,
    *,
    temperature_keV: np.ndarray | None = None,
) -> CurveSet:
    """Evaluate reactivities once and return backend-neutral log-log curve data."""
    reactions = dict(reactions) if reactions is not None else _default_reactivities()
    temperature = DEFAULT_TEMPERATURE_KEV if temperature_keV is None else np.asarray(temperature_keV, dtype=float)
    return CurveSet(
        [Curve(temperature, _evaluate(source, temperature), label, style={"linewidth": 2}) for label, source in reactions.items()],
        xlabel="Ion temperature [keV]",
        ylabel=r"$\langle \sigma v \rangle$ [m$^3$/s]",
        xscale="log",
        yscale="log",
    )


def _all_reactivity_curves(series: list[tuple[str, str, str, Any]], temperature_keV: np.ndarray) -> CurveSet:
    """Evaluate every registered parametrisation for the interactive explorer."""
    curves = []
    for reaction, source, label, relation in series:
        with np.errstate(all="ignore"):
            raw = np.asarray(relation.evaluate({"T_i": temperature_keV}), dtype=float)
        values = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 1e-40, None)
        curves.append(
            Curve(
                temperature_keV,
                values,
                label,
                style={"line_width": 2.2, "color": REACTION_COLORS.get(reaction, "#222222"), "line_dash": SOURCE_DASHES.get(source, ())},
                metadata={"reaction": reaction, "source": source},
            )
        )
    return CurveSet(curves, xlabel="Ion temperature [keV]", ylabel="⟨σv⟩ [m^3/s]", xscale="log", yscale="log")


def reactivity_app(
    *,
    x_limits: tuple[float, float] = (1e-3, 1.0e3),
    y_limits: tuple[float, float] = (1e-30, 1e-21),
    num_points: int = 1000,
    width: int = 960,
    height: int = 620,
) -> Any:
    """Return a standalone Bokeh layout for interactive reactivity exploration.

    Args:
        x_limits: Initial ion-temperature axis range in keV (log scale).
        y_limits: Initial reactivity axis range in m^3/s (log scale).
        num_points: Samples per curve across ``x_limits``.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        A Bokeh layout model suitable for ``show()`` or ``bokeh.embed.file_html()``.
    """
    from bokeh.models import CheckboxButtonGroup, LegendItem

    x_limits = validate_axis_limits(x_limits, label="x_limits")
    y_limits = validate_axis_limits(y_limits, label="y_limits")

    series = discover_reactivity_series()
    temperature_keV = log_grid(x_limits, num_points)

    plot = log_log_figure(
        x_limits=x_limits,
        y_limits=y_limits,
        width=width,
        height=height,
        title="Fusion Reactivities",
        x_label="Ion temperature [keV]",
        y_label="⟨σv⟩ [m^3/s]",
    )

    data = _all_reactivity_curves(series, temperature_keV)
    _plot, renderers, _sources = bokeh_curve_set(data, plot=plot, legend=False)
    legend_items = [LegendItem(label=curve.label, renderers=[renderer], visible=True) for curve, renderer in zip(data.curves, renderers, strict=True)]

    reaction_labels = sorted({reaction for reaction, *_ in series})
    source_labels = sorted({source for _, source, *_ in series}, key=_source_rank)
    reaction_selector = CheckboxButtonGroup(
        labels=reaction_labels, active=list(range(len(reaction_labels))), sizing_mode="stretch_width"
    )
    source_selector = CheckboxButtonGroup(
        labels=source_labels, active=list(range(len(source_labels))), sizing_mode="stretch_width"
    )

    layout, status = explorer_layout(
        plot,
        legend_items=legend_items,
        option_controls=(
            ("Reactions", reaction_selector),
            ("Sources", source_selector),
        ),
        x_limits=x_limits,
        y_limits=y_limits,
    )
    link_two_filter_visibility(
        reaction_selector,
        source_selector,
        reaction_labels,
        source_labels,
        [reaction for reaction, *_ in series],
        [source for _, source, *_ in series],
        renderers,
        legend_items,
        status,
    )

    return layout
