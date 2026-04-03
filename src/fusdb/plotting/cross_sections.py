"""Matplotlib helpers for reactor cross-section plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusdb.reactor_class import Reactor


def plot_cross_sections(reactor: "Reactor", *, ax=None, label: str | None = None):
    """Plot a plasma cross-section from the 95% flux-surface geometry.

    Args:
        reactor: Reactor supplying the geometry variables.
        ax: Optional Matplotlib axes to draw on.
        label: Optional legend label for the traced contour.

    Returns:
        The axes that received the plot.

    Raises:
        ValueError: If any required geometry variable is missing.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Read the 95% geometry variables from solved values first, then fall back
    # to the original YAML inputs when the reactor has not been solved yet.
    values: dict[str, object | None] = {}
    missing: list[str] = []
    for name in ("R", "a", "kappa_95", "delta_95"):
        var = reactor.variables_dict.get(name)
        value = None
        if var is not None:
            value = var.current_value
            if value is None:
                value = var.input_value
        values[name] = value
        if value is None:
            missing.append(name)

    if missing:
        raise ValueError(f"Missing 95% geometry variables for plotting: {', '.join(missing)}")
    R = float(values["R"])
    a = float(values["a"])
    kappa_95 = float(values["kappa_95"])
    delta_95 = float(values["delta_95"])

    # Build the shaped flux-surface contour in R-Z coordinates.
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    theta = np.linspace(0, 2 * np.pi, 200)
    r_vals = R + a * np.cos(theta + delta_95 * np.sin(theta))
    z_vals = kappa_95 * a * np.sin(theta)
    ax.plot(r_vals, z_vals, label=label or reactor.name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(True, alpha=0.3)
    return ax
