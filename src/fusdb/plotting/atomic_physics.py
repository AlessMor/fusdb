"""Atomic-physics rate-coefficient curves, ``<sigma v>`` versus temperature.

The AMJUEL rate relations -- H.2 fits mapping ``T_edge`` and H.4 fits mapping
``(n_e_edge, T_edge)`` to a single ``*_rate`` output -- are discovered from the
relation registry, so new fits appear without editing this module.
:func:`atomic_physics_app` builds a standalone, client-side interactive Bokeh
explorer with every rate as a toggleable curve, category/species filters, an
electron-density slider for the H.4 curves, and the AMJUEL fit errors in the
hover tooltip.

``bokeh`` is imported lazily inside the interactive builders, so it is *not*
pulled in by ``import fusdb.plotting``; install the ``plotting`` (or ``docs``)
extra to use the interactive app.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from fusdb.registry import RELATIONS

from ._bokeh import (
    axis_limit_controls,
    labeled_row,
    link_two_filter_visibility,
    log_grid,
    log_log_figure,
    model_html,
    validate_axis_limits,
    write_html,
)

# Display order of the process categories (relation subpackage names) and the
# species filter buttons (hydrogenic first, so the default selection is leading).
CATEGORY_ORDER = (
    "elastic_scattering",
    "charge_exchange",
    "ionization",
    "recombination",
    "molecular_ionization",
    "dissociation",
    "dissociative_attachment",
    "dissociative_ionization",
    "dissociative_excitation",
    "dissociative_recombination",
    "mar_via_h2_plus",
    "mar_via_h_minus",
)
SPECIES_ORDER = (
    "H",
    "H2",
    "H2+",
    "He",
    "He+",
    "N2",
    "N2+",
    "B",
    "B+",
    "Be",
    "Be+",
    "C",
    "C+",
    "N",
    "O",
    "O+",
    "Ne",
    "Ne+",
    "Ar",
    "Ar+",
    "Kr",
    "Xe",
    "Fe",
    "Fe+",
    "W",
)
HYDROGENIC_SPECIES = ("H", "H2", "H2+")

# Stable colour per process family. Categories that are channels of the same
# family share a hue (curve identity comes from hover, legend and filters).
CATEGORY_COLORS = {
    "recombination": "#2a78d6",
    "ionization": "#1baf7a",
    "elastic_scattering": "#2870b9",
    "mar_via_h2_plus": "#008300",
    "mar_via_h_minus": "#008300",
    "dissociative_ionization": "#4a3aa7",
    "dissociative_excitation": "#8a63c7",
    "molecular_ionization": "#4a3aa7",
    "dissociation": "#e34948",
    "dissociative_attachment": "#b85a21",
    "charge_exchange": "#e87ba4",
    "dissociative_recombination": "#eb6834",
}

# The electron-density slider covers the AMJUEL H.4 fit validity range, one
# curve per decade; ``y<exponent>`` columns are precomputed for each of these.
DENSITY_EXPONENTS = tuple(range(14, 23))
DEFAULT_DENSITY_EXPONENT = 19

# Display floor/ceiling applied to the precomputed curves: keeps the log axis
# finite where a fit is evaluated outside its temperature validity range.
_RATE_FLOOR = 1e-35
_RATE_CEILING = 1.0


class RateSeries(NamedTuple):
    """One rate-coefficient curve discovered from the registry."""

    category: str
    species: str
    label: str
    reaction: str
    fit_error: str
    density_dependent: bool
    relation: Any


def _is_rate_relation(relation: Any) -> bool:
    """Return whether a relation maps ``T_edge`` (+ optional ``n_e_edge``) to one ``*_rate``."""
    outputs = relation.outputs
    return (
        "atomic_physics" in relation.tags
        and set(relation.input_names) in ({"T_edge"}, {"T_edge", "n_e_edge"})
        and len(outputs) == 1
        and outputs[0].endswith("_rate")
    )


def _species_label(module_stem: str) -> str:
    """Return the display species for a relation module stem (``H2_plus`` -> ``H2+``)."""
    return module_stem.replace("_plus", "+").replace("_minus", "-")


def _doc_line(relation: Any, prefix: str) -> str:
    """Return the content after ``prefix`` on the matching docstring line, if any."""
    for line in (relation.func.__doc__ or "").splitlines():
        line = line.strip()
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().rstrip(".")
    return ""


def _order(sequence: tuple[str, ...], value: str) -> int:
    return sequence.index(value) if value in sequence else len(sequence)


def discover_rate_series() -> list[RateSeries]:
    """Return every atomic-physics rate curve discovered from the registry.

    Category and species come from the defining module path
    (``fusdb.relations.atomic_physics.<category>.<species>``); the label is the
    relation name without the ``"AMJUEL H.<n> "``/``" rate"`` affixes. Sorted by
    :data:`CATEGORY_ORDER`, then :data:`SPECIES_ORDER`, then label.
    """
    series: list[RateSeries] = []
    for relation in RELATIONS:
        if not _is_rate_relation(relation):
            continue
        module_parts = relation.func.__module__.split(".")
        category = module_parts[-2] if len(module_parts) >= 2 else "other"
        species = _species_label(module_parts[-1])
        label = re.sub(r"^AMJUEL H\.\d+ ", "", relation.name).removesuffix(" rate")
        series.append(
            RateSeries(
                category,
                species,
                label,
                _doc_line(relation, "Reaction:"),
                _doc_line(relation, "Fit error (AMJUEL):"),
                "n_e_edge" in relation.input_names,
                relation,
            )
        )
    return sorted(
        series,
        key=lambda item: (_order(CATEGORY_ORDER, item.category), _order(SPECIES_ORDER, item.species), item.label),
    )


def _rate_curve(relation: Any, temperature_keV: np.ndarray, n_e_m3: float | None = None) -> np.ndarray:
    """Return one clipped rate column (H.4 relations also take a density)."""
    # ``evaluate`` returns raw fit values (no constraint enforcement) under a
    # strict errstate that raises on overflow, so points where a fit is taken
    # outside its temperature validity are recovered one by one (an overflowing
    # exponent means off-scale high) and clipped to a finite band for the log axis.
    def _inputs(T: Any) -> dict[str, Any]:
        return {"T_edge": T} if n_e_m3 is None else {"T_edge": T, "n_e_edge": n_e_m3}

    try:
        raw = np.asarray(relation.evaluate(_inputs(temperature_keV)), dtype=float)
    except FloatingPointError:
        raw = np.empty(temperature_keV.shape, dtype=float)
        for index, T_keV in enumerate(temperature_keV):
            try:
                raw[index] = float(relation.evaluate(_inputs(T_keV)))
            except FloatingPointError:
                raw[index] = _RATE_CEILING
    cleaned = np.nan_to_num(raw, nan=_RATE_FLOOR, posinf=_RATE_CEILING, neginf=_RATE_FLOOR)
    return np.clip(cleaned, _RATE_FLOOR, _RATE_CEILING).astype(np.float32)


def atomic_physics_app(
    *,
    x_limits: tuple[float, float] = (0.1, 2.0e4),
    y_limits: tuple[float, float] = (1e-22, 1e-12),
    num_points: int = 250,
    width: int = 960,
    height: int = 620,
) -> Any:
    """Return a standalone Bokeh layout for interactive rate exploration.

    Args:
        x_limits: Initial electron-temperature axis range in eV (log scale).
        y_limits: Initial rate-coefficient axis range in m^3/s (log scale).
        num_points: Samples per curve across ``x_limits``.
        width: Plot width in pixels.
        height: Plot height in pixels.

    Returns:
        A Bokeh layout model (embed with :func:`render_atomic_physics_app_html`).
    """
    from bokeh.layouts import column
    from bokeh.models import (
        CheckboxButtonGroup,
        ColumnDataSource,
        CustomJS,
        HoverTool,
        Legend,
        LegendItem,
        Slider,
    )

    x_limits = validate_axis_limits(x_limits, label="x_limits")
    y_limits = validate_axis_limits(y_limits, label="y_limits")

    series = discover_rate_series()
    temperature_eV = log_grid(x_limits, num_points)
    temperature_keV = temperature_eV / 1.0e3

    plot = log_log_figure(
        x_limits=x_limits,
        y_limits=y_limits,
        width=width,
        height=height,
        title="Atomic & Molecular Rate Coefficients (AMJUEL H.2 & H.4)",
        x_label="Electron temperature [eV]",
        y_label="⟨σv⟩ [m^3/s]",
    )

    category_labels = sorted({item.category for item in series}, key=lambda c: _order(CATEGORY_ORDER, c))
    species_labels = sorted({item.species for item in series}, key=lambda s: _order(SPECIES_ORDER, s))
    initial_species = {label for label in species_labels if label in HYDROGENIC_SPECIES} or set(species_labels)

    renderers = []
    density_sources = []
    legend_items: list = []
    for item in series:
        if item.density_dependent:
            columns = {
                f"y{exponent}": _rate_curve(item.relation, temperature_keV, n_e_m3=10.0**exponent)
                for exponent in DENSITY_EXPONENTS
            }
            columns["y"] = columns[f"y{DEFAULT_DENSITY_EXPONENT}"]
        else:
            columns = {"y": _rate_curve(item.relation, temperature_keV)}
        source = ColumnDataSource({"x": temperature_eV.astype(np.float32), **columns})
        hover_name = " — ".join(
            part
            for part in (
                item.label,
                item.reaction,
                "H.4 fit at slider density" if item.density_dependent else "",
                f"fit err {item.fit_error}" if item.fit_error else "",
            )
            if part
        )
        visible = item.species in initial_species
        renderer = plot.line(
            "x",
            "y",
            source=source,
            name=hover_name,
            line_width=2.0,
            color=CATEGORY_COLORS.get(item.category, "#222222"),
            visible=visible,
        )
        if item.density_dependent:
            density_sources.append(source)
        renderers.append(renderer)
        legend_label = f"{item.label} [H.4]" if item.density_dependent else item.label
        legend_items.append(LegendItem(label=legend_label, renderers=[renderer], visible=visible))

    plot.add_layout(
        Legend(items=legend_items, click_policy="hide", label_text_font_size="8pt", spacing=0), "right"
    )
    plot.add_tools(HoverTool(tooltips=[("", "$name"), ("T_e", "$x eV"), ("rate", "$y m^3/s")], line_policy="nearest"))

    category_selector = CheckboxButtonGroup(
        labels=[label.replace("_", " ") for label in category_labels],
        active=list(range(len(category_labels))),
        sizing_mode="stretch_width",
    )
    species_selector = CheckboxButtonGroup(
        labels=species_labels,
        active=[index for index, label in enumerate(species_labels) if label in initial_species],
        sizing_mode="stretch_width",
    )
    density_slider = Slider(
        start=DENSITY_EXPONENTS[0],
        end=DENSITY_EXPONENTS[-1],
        value=DEFAULT_DENSITY_EXPONENT,
        step=1,
        title="Electron density log10(n_e [m^-3]), applies to the [H.4] curves",
        sizing_mode="stretch_width",
    )
    density_slider.js_on_change(
        "value",
        CustomJS(
            args=dict(sources=density_sources, slider=density_slider),
            code="""
const key = "y" + slider.value;
for (const source of sources) {
  source.data["y"] = source.data[key];
  source.change.emit();
}
""",
        ),
    )
    limit_widgets, status = axis_limit_controls(plot, x_limits, y_limits)
    link_two_filter_visibility(
        category_selector,
        species_selector,
        category_labels,
        species_labels,
        [item.category for item in series],
        [item.species for item in series],
        renderers,
        legend_items,
        status,
    )

    return column(
        plot,
        labeled_row("Density", density_slider),
        labeled_row("Processes", category_selector),
        labeled_row("Species", species_selector),
        labeled_row("Limits", *limit_widgets),
        sizing_mode="stretch_width",
    )


def render_atomic_physics_app_html(*, title: str = "Atomic Physics Rate Plotter", **kwargs: Any) -> str:
    """Return a self-contained interactive HTML document (BokehJS from CDN).

    ``**kwargs`` are forwarded to :func:`atomic_physics_app`.
    """
    return model_html(atomic_physics_app(**kwargs), title)


def save_atomic_physics_app_html(path: str | Path, **kwargs: Any) -> Path:
    """Write the interactive atomic-physics plotter HTML to ``path`` and return it."""
    return write_html(path, render_atomic_physics_app_html(**kwargs))
