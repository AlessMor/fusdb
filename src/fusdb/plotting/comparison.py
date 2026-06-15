"""Grouped-bar comparison of scalar metrics across cases.

Generalises the operational-limit utilisation bars in the profile-optimisation
example and is reusable for any per-case scalar comparison (e.g. comparing the
same metrics across several reactors).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from matplotlib.axes import Axes

from .style import axes, color_cycle


def plot_metric_comparison(
    cases: Mapping[str, Mapping[str, float]],
    *,
    ax: Axes | None = None,
    reference: float | None = None,
    annotate: bool = True,
    ylabel: str | None = None,
    title: str | None = None,
    **bar_kw: Any,
) -> Axes:
    """Compare scalar metrics across named cases as grouped bars.

    Args:
        cases: Mapping of case label -> ``{metric: value}``. The union of metric
            names (first-seen order) defines the x groups.
        reference: Optional horizontal reference line (e.g. ``1.0`` for limits).
        annotate: When ``True``, label each bar with its value.
        ylabel, title: Optional axis label and title.
        **bar_kw: Forwarded to ``Axes.bar``.

    Returns:
        The axis the bars were drawn on.
    """
    case_labels = list(cases)
    metrics: list[str] = []
    for values in cases.values():
        metrics.extend(metric for metric in values if metric not in metrics)

    ax = axes(ax, figsize=(1.6 * len(metrics) + 3, 4.2))
    colors = color_cycle(case_labels)
    positions = np.arange(len(metrics))
    width = 0.8 / max(len(case_labels), 1)

    for index, case in enumerate(case_labels):
        offset = (index - (len(case_labels) - 1) / 2) * width
        heights = [float(cases[case].get(metric, np.nan)) for metric in metrics]
        bars = ax.bar(positions + offset, heights, width, label=case, color=colors[case], **bar_kw)
        if annotate:
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    if reference is not None:
        ax.axhline(reference, color="red", lw=1.5, ls="--", label="limit")

    ax.set_xticks(positions)
    ax.set_xticklabels(metrics)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax
