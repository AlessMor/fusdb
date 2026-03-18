"""Matplotlib helpers for POPCON-style scans."""

from __future__ import annotations


def plot_popcon(
    result: dict[str, object],
    *,
    x: str,
    y: str,
    fill: str,
    contours: list[str] | None = None,
    contour_levels: dict[str, list[float]] | None = None,
    contour_counts: dict[str, int] | None = None,
    constraint_contours: bool = True,
    slice: dict[str, int | float] | None = None,
    reduce: dict[str, str] | None = None,
    best: dict[str, str] | None = None,
    ax=None,
):
    """Plot POPCON results with masked fill and contour overlays."""
    import matplotlib.pyplot as plt
    import numpy as np

    axes = result.get("axes", {})
    axis_order = result.get("axis_order", list(axes.keys()))
    outputs = result.get("outputs", {})
    margins = result.get("margins", {})
    allowed = result.get("allowed")

    if x not in axes or y not in axes:
        raise ValueError("x and y must be scan axes present in result['axes'].")

    if allowed is None:
        raise ValueError("Result missing 'allowed' mask.")

    x_vals = axes[x]
    y_vals = axes[y]

    def _select_2d(arr: np.ndarray, *, label: str) -> np.ndarray:
        data = np.asarray(arr)
        if data.ndim == 0:
            return np.broadcast_to(data, (len(x_vals), len(y_vals)))
        if data.ndim == 1:
            if data.shape[0] == len(x_vals):
                return np.broadcast_to(data[:, None], (len(x_vals), len(y_vals)))
            if data.shape[0] == len(y_vals):
                return np.broadcast_to(data[None, :], (len(x_vals), len(y_vals)))
            raise ValueError(f"{label} length must match x or y axis for plotting.")
        if data.ndim < 2:
            raise ValueError(f"{label} must be at least 2D for plotting.")

        if data.ndim > 2:
            if slice is None and reduce is None:
                raise ValueError("Provide slice or reduce for scans with >2 dimensions.")

            if slice is not None:
                indexers = []
                for name in axis_order:
                    if name in (x, y):
                        indexers.append(slice(None))
                        continue
                    if name not in slice:
                        raise ValueError(f"Missing slice for axis '{name}'.")
                    selector = slice[name]
                    axis_vals = axes[name]
                    if isinstance(selector, int):
                        indexers.append(selector)
                    else:
                        idx = int(np.argmin(np.abs(axis_vals - float(selector))))
                        indexers.append(idx)
                data = data[tuple(indexers)]
            else:
                metric_name = reduce.get("metric") if reduce else None
                mode = reduce.get("mode", "max") if reduce else "max"
                metric = outputs.get(metric_name) if metric_name else None
                if metric is None:
                    raise ValueError("reduce requires a metric present in outputs.")

                perm = [axis_order.index(x), axis_order.index(y)]
                perm += [axis_order.index(name) for name in axis_order if name not in (x, y)]
                data_r = np.moveaxis(data, perm, range(len(perm)))
                metric_r = np.moveaxis(np.asarray(metric), perm, range(len(perm)))
                allowed_r = np.moveaxis(np.asarray(allowed), perm, range(len(perm)))

                flat_data = data_r.reshape(data_r.shape[0], data_r.shape[1], -1)
                flat_metric = metric_r.reshape(metric_r.shape[0], metric_r.shape[1], -1)
                flat_allowed = allowed_r.reshape(allowed_r.shape[0], allowed_r.shape[1], -1)

                if mode == "min":
                    masked = np.where(flat_allowed, flat_metric, np.inf)
                    best_idx = np.nanargmin(masked, axis=2)
                else:
                    masked = np.where(flat_allowed, flat_metric, -np.inf)
                    best_idx = np.nanargmax(masked, axis=2)

                data = np.take_along_axis(flat_data, best_idx[..., None], axis=2)[..., 0]

        remaining = [name for name in axis_order if name in (x, y)]
        if remaining != [x, y]:
            data = np.moveaxis(data, [remaining.index(x), remaining.index(y)], [0, 1])

        return data

    def _as_float_array(arr: np.ndarray) -> np.ndarray:
        if arr.dtype != object:
            return arr
        out = np.empty(arr.shape, dtype=float)
        it = np.nditer(arr, flags=["multi_index", "refs_ok"])
        for item in it:
            try:
                out[it.multi_index] = float(item.item())
            except Exception:
                out[it.multi_index] = np.nan
        return out

    fill_data = outputs.get(fill)
    if fill_data is None:
        raise ValueError(f"Fill variable '{fill}' not found in outputs.")

    fill_2d = _select_2d(_as_float_array(np.asarray(fill_data)), label=fill)
    allowed_2d = _select_2d(np.asarray(allowed), label="allowed")

    masked_fill = np.ma.masked_where(~allowed_2d, fill_2d)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    def _plot_ready(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.shape == (len(x_vals), len(y_vals)):
            return arr.T
        return arr

    mesh = ax.pcolormesh(x_vals, y_vals, _plot_ready(masked_fill), shading="auto", cmap="viridis")
    plt.colorbar(mesh, ax=ax, label=fill)

    legend_handles = []
    legend_labels = []
    pretty_names = {"Q_sci": "Q", "n_GW": "n_greenwald"}

    contour_colors = [
        "#d62728",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#a52a2a",
        "#4d4d4d",
        "#b15928",
    ]
    color_cycle = contour_colors
    color_idx = 0

    def _next_color() -> tuple[float, float, float]:
        nonlocal color_idx
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1
        return color

    def _label_values(cs):
        if not cs.levels.size:
            return
        ax.clabel(
            cs,
            fmt=lambda val: f"{val:g}",
            inline=True,
            fontsize=8,
        )

    if constraint_contours:
        for name, margin in margins.items():
            if margin is None:
                continue
            margin_2d = _select_2d(_as_float_array(np.asarray(margin)), label=name)
            color = _next_color()
            cs = ax.contour(
                x_vals,
                y_vals,
                _plot_ready(margin_2d),
                levels=[0.0],
                colors=[color],
                linewidths=1.0,
            )
            _label_values(cs)
            from matplotlib.lines import Line2D

            legend_handles.append(Line2D([], [], color=color, linewidth=1.0))
            legend_labels.append(pretty_names.get(name, name))

    if contours:
        for name in contours:
            data = outputs.get(name)
            if data is None:
                continue
            if name in ("n_GW", "n_greenwald"):
                try:
                    val = float(np.asarray(data, dtype=float).reshape(-1)[0])
                except Exception:
                    continue
                color = _next_color()
                from matplotlib.lines import Line2D

                if y in ("n_avg", "n_e"):
                    ax.axhline(val, color=color, linewidth=1.2)
                    ax.text(x_vals[len(x_vals) // 2], val, f"{val:g}", color=color, fontsize=8, va="bottom")
                elif x in ("n_avg", "n_e"):
                    ax.axvline(val, color=color, linewidth=1.2)
                    ax.text(val, y_vals[len(y_vals) // 2], f"{val:g}", color=color, fontsize=8, ha="left")
                legend_handles.append(Line2D([], [], color=color, linewidth=1.2))
                legend_labels.append(pretty_names.get(name, name))
                continue
            data_2d = _select_2d(_as_float_array(np.asarray(data)), label=name)
            color = _next_color()
            levels = None
            if contour_levels and name in contour_levels:
                levels = contour_levels[name]
            elif contour_counts and name in contour_counts:
                try:
                    levels = int(contour_counts[name])
                except Exception:
                    levels = None
            finite = np.isfinite(data_2d)
            is_constant = False
            if finite.any():
                vmin = float(np.nanmin(data_2d))
                vmax = float(np.nanmax(data_2d))
                scale = max(abs(vmin), abs(vmax), 1.0)
                is_constant = abs(vmax - vmin) <= 1e-12 * scale
            if levels is None and is_constant and fill_2d is not None:
                const_val = float(np.nanmean(data_2d))
                cs = ax.contour(
                    x_vals,
                    y_vals,
                    _plot_ready(fill_2d),
                    levels=[const_val],
                    colors=[color],
                    linewidths=0.9,
                    alpha=0.9,
                )
                _label_values(cs)
                from matplotlib.lines import Line2D

                legend_handles.append(Line2D([], [], color=color, linewidth=1.0))
                legend_labels.append(pretty_names.get(name, name))
                continue
            cs = ax.contour(
                x_vals,
                y_vals,
                _plot_ready(data_2d),
                levels=levels,
                colors=[color],
                linewidths=0.9,
                alpha=0.9,
            )
            _label_values(cs)
            from matplotlib.lines import Line2D

            legend_handles.append(Line2D([], [], color=color, linewidth=1.0))
            legend_labels.append(pretty_names.get(name, name))

    if best:
        metric_name = best.get("metric")
        mode = best.get("mode", "max")
        metric = outputs.get(metric_name)
        if metric is not None:
            metric_2d = _select_2d(_as_float_array(np.asarray(metric)), label=metric_name)
            metric_masked = np.where(allowed_2d, metric_2d, np.nan)
            if np.all(np.isnan(metric_masked)):
                return ax
            if mode == "min":
                idx = np.nanargmin(metric_masked)
            else:
                idx = np.nanargmax(metric_masked)
            ix, iy = np.unravel_index(idx, metric_masked.shape)
            ax.scatter([x_vals[ix]], [y_vals[iy]], color="red", s=30, zorder=5)
            ax.annotate(
                f"{metric_name}={metric_2d[ix, iy]:.3g}",
                (x_vals[ix], y_vals[iy]),
                textcoords="offset points",
                xytext=(6, 6),
                color="red",
            )

    if legend_handles:
        ax.legend(legend_handles, legend_labels, title="Contours", loc="best", fontsize=8)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"POPCON: {fill}")
    ax.grid(True, alpha=0.3)
    return ax
