"""2-D parameter maps: a filled field with optional iso-contours and a marker.

Reusable form of the ``beta utilisation vs peaking`` map in
``examples/dhe3_profile_shape_optimization.ipynb`` (a filled field, a fixed
fusion-power iso-line, and the optimum marker).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes

from .style import axes


def plot_parameter_map(
    x: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    z: Sequence[Sequence[float]],
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    levels: int = 20,
    clabel: str | None = None,
    iso_field: Sequence[Sequence[float]] | None = None,
    iso_levels: Sequence[float] | None = None,
    iso_label: str | None = None,
    marker: tuple[float, float] | None = None,
    marker_label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Filled contour of ``z`` over the ``(x, y)`` mesh.

    Args:
        x, y, z: 2-D meshgrid arrays; ``z`` is the filled field.
        cmap, levels: Colormap and number of filled contour levels.
        clabel: Colorbar label.
        iso_field, iso_levels: Optional second field drawn as iso-contours at the
            given levels (e.g. a fixed fusion-power line over a utilisation map).
        iso_label: Optional inline label format for the iso-contours.
        marker: Optional ``(x, y)`` point highlighted with a star.
        marker_label: Legend label for the marker.
        xlabel, ylabel, title: Optional axis labels and title.

    Returns:
        The axis the map was drawn on.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    ax = axes(ax, figsize=(7, 5.2))
    filled = ax.contourf(x, y, z, levels=levels, cmap=cmap)
    ax.figure.colorbar(filled, ax=ax, label=clabel)

    if iso_field is not None and iso_levels is not None:
        iso = ax.contour(x, y, np.asarray(iso_field, dtype=float), levels=list(iso_levels),
                         colors="k", linewidths=2)
        if iso_label:
            ax.clabel(iso, fmt=iso_label, fontsize=8)

    if marker is not None:
        ax.plot(marker[0], marker[1], "r*", ms=16, label=marker_label or "optimum")
        ax.legend(loc="upper right")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax
