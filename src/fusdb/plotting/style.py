"""Shared axis helpers and a common palette for the fusdb plotters."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Colour-blind friendly qualitative palette (Dark2) reused across plotters.
RELATION_COLOR = "#d95f02"
VARIABLE_COLOR = "#1b9e77"
PALETTE = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d")


def axes(ax: Axes | None = None, *, figsize: tuple[float, float] | None = None) -> Axes:
    """Return an axis to draw on, creating a new figure when ``ax`` is ``None``."""
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=figsize)
    return ax
