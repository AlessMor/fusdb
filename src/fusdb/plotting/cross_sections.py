"""Matplotlib helpers for reactor cross-section plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusdb.reactor_class import Reactor


def plot_cross_sections(reactor: "Reactor", *, ax=None, label: str | None = None):
    """Plot a plasma cross-section using 95% flux-surface geometry."""
    import matplotlib.pyplot as plt
    import numpy as np

    values = {}
    for name in ("R", "a", "kappa_95", "delta_95"):
        var = reactor.variables_dict.get(name)
        values[name] = None if var is None else var.current_value if var.current_value is not None else var.input_value
    R, a, kappa_95, delta_95 = (
        values["R"],
        values["a"],
        values["kappa_95"],
        values["delta_95"],
    )

    if any(val is None for val in (R, a, kappa_95, delta_95)):
        missing = []
        if R is None:
            missing.append("R")
        if a is None:
            missing.append("a")
        if kappa_95 is None:
            missing.append("kappa_95")
        if delta_95 is None:
            missing.append("delta_95")
        raise ValueError(f"Missing 95% geometry variables for plotting: {', '.join(missing)}")

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
