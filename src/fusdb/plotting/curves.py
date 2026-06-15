"""Generic x-y line overlays with optional point markers.

Covers parametric scans and consistency plots such as the ``tau_E`` vs
``P_loss`` figure in ``examples/tau_E_solver.ipynb`` (energy-balance and scaling
curves with an analytical-intersection / solved-point marker).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes

from .style import axes

XY = Tuple[Sequence[float], Sequence[float]]


def plot_curves(
    curves: Mapping[str, XY],
    *,
    ax: Axes | None = None,
    markers: Mapping[str, XY] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    **plot_kw: Any,
) -> Axes:
    """Overlay labelled line curves, optionally with standalone point markers.

    Args:
        curves: Mapping of label -> ``(x, y)`` drawn as connected lines.
        markers: Mapping of label -> ``(x, y)`` drawn as points (e.g. an
            analytical intersection or a solved operating point).
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        title: Optional axis title.
        xscale: matplotlib x-axis scale, e.g. ``"log"``.
        yscale: matplotlib y-axis scale, e.g. ``"log"``.
        **plot_kw: Forwarded to ``Axes.plot`` for the line curves.

    Returns:
        The axis the curves were drawn on.
    """
    ax = axes(ax, figsize=(10, 6))
    plot_kw.setdefault("linewidth", 2)
    for label, (x, y) in curves.items():
        ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), label=label, **plot_kw)
    for label, (x, y) in (markers or {}).items():
        ax.plot(np.atleast_1d(x), np.atleast_1d(y), "o", markersize=8, label=label)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
