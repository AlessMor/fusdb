"""Radial plasma-profile plots versus the normalised minor radius ``rho``.

Generalises the density/temperature panels in the profile-optimisation example
(``examples/dhe3_profile_shape_optimization.ipynb``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import axes


def plot_profiles(
    rho: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    ax: Axes | None = None,
    normalize: float | None = None,
    xlabel: str = r"$\rho$",
    ylabel: str | None = None,
    title: str | None = None,
    **plot_kw: Any,
) -> Axes:
    """Plot one or more radial profiles on a shared axis.

    Args:
        rho: Normalised minor-radius grid (0..1).
        series: Mapping of label -> profile values aligned with ``rho``.
        normalize: Optional divisor applied to every series (e.g. ``1e20`` to
            show density in units of ``1e20 m^-3``).
        ax: Existing axis to draw on; a new figure is created when omitted.
        **plot_kw: Forwarded to ``Axes.plot``.

    Returns:
        The axis the profiles were drawn on.
    """
    rho = np.asarray(rho, dtype=float)
    scale = normalize or 1.0
    ax = axes(ax, figsize=(7, 4.5))
    plot_kw.setdefault("linewidth", 2)
    for label, values in series.items():
        ax.plot(rho, np.asarray(values, dtype=float) / scale, label=label, **plot_kw)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_profile_grid(
    rho: Sequence[float],
    panels: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    figsize: tuple[float, float] | None = None,
    ylabels: Mapping[str, str] | None = None,
) -> Figure:
    """Plot several profile groups side by side, one subplot per panel.

    Args:
        rho: Normalised minor-radius grid shared by every panel.
        panels: Mapping of panel title -> ``series`` mapping (as accepted by
            :func:`plot_profiles`).
        figsize: Optional explicit figure size; defaults scale with panel count.
        ylabels: Optional mapping of panel title -> y-axis label.

    Returns:
        The created figure.
    """
    ylabels = ylabels or {}
    columns = max(len(panels), 1)
    fig, axs = plt.subplots(1, columns, figsize=figsize or (6 * columns, 4.2), squeeze=False)
    for ax, (title, series) in zip(axs[0], panels.items()):
        plot_profiles(rho, series, ax=ax, title=title, ylabel=ylabels.get(title))
    fig.tight_layout()
    return fig
