"""POPCON plots: contour maps of a 2-D popcon-mode scan result.

Draws the classic plasma-operating-contour figure from the ``"popcon"``
payload produced by :mod:`fusdb.modes.popcon`: an optional filled field, one
labelled iso-contour set per requested output, and a grey mask over grid
points that did not solve.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from ..registry import VARIABLES
from .style import axes, color_cycle


def _field_label(name: str) -> str:
    """``name [unit]`` from the registry; plain ``name`` for ad-hoc fields."""
    if name in VARIABLES:
        unit = VARIABLES.get(name).unit
        if unit and unit != "dimensionless":
            return f"{name} [{unit}]"
    return name


def plot_popcon(
    result: Mapping[str, Any],
    *,
    fill: str | None = None,
    contours: Sequence[str] = (),
    levels: Mapping[str, Sequence[float]] | None = None,
    fill_levels: int | Sequence[float] = 20,
    cmap: str = "viridis",
    colors: Mapping[str, str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a popcon scan as filled/line contours over the two axis variables.

    Args:
        result: A popcon mode result dict, or its ``"popcon"`` payload.
        fill: Optional field drawn as a filled contour with a colorbar.
        contours: Fields drawn as labelled iso-contour lines.
        levels: Optional explicit line-contour levels per field name.
        fill_levels: Filled contour levels for ``fill``: a count or a sequence.
        cmap: Colormap for the filled field.
        colors: Optional per-contour line colour, keyed by field name;
            fields absent from the mapping fall back to the default palette.
        ax: Optional axis to draw on.
        title: Optional axis title.

    Returns:
        The axis the popcon was drawn on.
    """
    payload = result.get("popcon", result)
    if fill is None and not contours:
        available = ", ".join(sorted(payload["fields"]))
        raise ValueError(f"Pass fill= and/or contours=; available fields: {available}.")
    x = np.asarray(payload["x"]["values"], dtype=float)
    y = np.asarray(payload["y"]["values"], dtype=float)
    fields = payload["fields"]
    success = np.asarray(payload["success"], dtype=bool)
    levels = dict(levels or {})

    def field(name: str) -> np.ndarray:
        try:
            return np.asarray(fields[name], dtype=float)
        except KeyError:
            available = ", ".join(sorted(fields))
            raise KeyError(f"Field {name!r} not in the scan; available: {available}.") from None

    ax = axes(ax, figsize=(7, 5.2))
    if fill is not None:
        filled = ax.contourf(x, y, field(fill), levels=fill_levels, cmap=cmap)
        ax.figure.colorbar(filled, ax=ax, label=_field_label(fill))

    handles = []
    palette = color_cycle(contours)
    overrides = dict(colors or {})
    for name in contours:
        line_color = overrides.get(name, palette[name])
        contour_kwargs = {"colors": line_color, "linewidths": 1.5}
        if levels.get(name) is not None:
            contour_kwargs["levels"] = list(levels[name])
        contour = ax.contour(x, y, field(name), **contour_kwargs)
        ax.clabel(contour, fmt="%g", fontsize=8)
        handles.append(Line2D([], [], color=line_color, label=_field_label(name)))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    if not success.all():
        # Grey out the unsolved (infeasible) region.
        ax.contourf(x, y, (~success).astype(float), levels=[0.5, 1.5], colors=["#9e9e9e"], alpha=0.45)

    ax.set_xlabel(_field_label(payload["x"]["name"]))
    ax.set_ylabel(_field_label(payload["y"]["name"]))
    if title:
        ax.set_title(title)
    return ax
