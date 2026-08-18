"""Generic Matplotlib and Bokeh renderers for backend-neutral plot data."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

from .data import CurveSet, FieldMap
from .style import axes


def plot_curve_set(data: CurveSet, *, ax: Axes | None = None, **plot_kw: Any) -> Axes:
    """Render a :class:`~fusdb.plotting.data.CurveSet` with Matplotlib."""
    ax = axes(ax, figsize=(10, 6))
    for curve in data.curves:
        style = {**plot_kw, **curve.style}
        marker_only = style.pop("marker_only", False)
        if marker_only:
            style.setdefault("linestyle", "None")
            style.setdefault("marker", "o")
        ax.plot(curve.x, curve.y, label=curve.label, **style)
    ax.set_xscale(data.xscale)
    ax.set_yscale(data.yscale)
    if data.xlabel:
        ax.set_xlabel(data.xlabel)
    if data.ylabel:
        ax.set_ylabel(data.ylabel)
    if data.title:
        ax.set_title(data.title)
    ax.grid(True, which="both" if "log" in (data.xscale, data.yscale) else "major", alpha=0.3)
    if data.curves:
        ax.legend()
    return ax


def plot_field_map(
    data: FieldMap,
    *,
    fill: str | None = None,
    contours: tuple[str, ...] = (),
    levels: dict[str, Any] | None = None,
    fill_levels: int | Any = 20,
    cmap: str = "viridis",
    colors: dict[str, str] | None = None,
    contour_label: str = "%g",
    ax: Axes | None = None,
) -> Axes:
    """Render a filled/contoured :class:`FieldMap` with Matplotlib."""
    from matplotlib.lines import Line2D

    if fill is None and not contours:
        raise ValueError("Pass fill= and/or contours=.")
    if fill is not None and fill not in data.fields:
        raise KeyError(f"Field {fill!r} is not present.")
    levels = levels or {}
    field_labels = data.metadata.get("field_labels", {})
    ax = axes(ax, figsize=(7, 5.2))
    if fill is not None:
        image = ax.contourf(data.x, data.y, data.fields[fill], levels=fill_levels, cmap=cmap)
        ax.figure.colorbar(image, ax=ax, label=field_labels.get(fill, fill))
    handles = []
    palette = tuple(colors.values()) if colors else ()
    for index, name in enumerate(contours):
        if name not in data.fields:
            raise KeyError(f"Field {name!r} is not present.")
        color = (colors or {}).get(name, palette[index % len(palette)] if palette else None)
        kwargs = {"linewidths": 1.5}
        if color:
            kwargs["colors"] = color
        if name in levels:
            kwargs["levels"] = levels[name]
        contour = ax.contour(data.x, data.y, data.fields[name], **kwargs)
        ax.clabel(contour, fmt=contour_label, fontsize=8)
        handles.append(Line2D([], [], color=color or "#222222", label=field_labels.get(name, name)))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    if data.xlabel:
        ax.set_xlabel(data.xlabel)
    if data.ylabel:
        ax.set_ylabel(data.ylabel)
    if data.title:
        ax.set_title(data.title)
    mask = data.metadata.get("mask")
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != data.x.shape:
            raise ValueError(f"FieldMap mask has shape {mask_array.shape}; expected {data.x.shape}.")
        ax.contourf(data.x, data.y, mask_array.astype(float), levels=[0.5, 1.5], colors=["#9e9e9e"], alpha=0.45)
    return ax
