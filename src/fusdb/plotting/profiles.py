"""Radial plasma-profile plots versus the normalised minor radius ``rho``.

Generalises the density/temperature panels in the profile-optimisation example
(``examples/dhe3_profile_shape_optimization.ipynb``).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .data import Curve, CurveSet


def profile_curves(
    rho: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    normalize: float | None = None,
    xlabel: str = r"$\rho$",
    ylabel: str | None = None,
    title: str | None = None,
) -> CurveSet:
    """Prepare backend-neutral radial-profile curve data.

    Args:
        rho: Normalised minor-radius grid (0..1).
        series: Mapping of label -> profile values aligned with ``rho``.
        normalize: Optional divisor applied to every series (e.g. ``1e20`` to
            show density in units of ``1e20 m^-3``).
    Render the result with :func:`fusdb.plotting.plot_curve_set` or
    :func:`fusdb.plotting.bokeh_curve_set`.
    """
    rho = np.asarray(rho, dtype=float)
    scale = normalize or 1.0
    return CurveSet(
        [Curve(rho, np.asarray(values, dtype=float) / scale, label, style={"linewidth": 2}) for label, values in series.items()],
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
