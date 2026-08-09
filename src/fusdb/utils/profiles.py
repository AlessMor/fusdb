"""Numerical integration and averaging helpers for one-dimensional profiles.

``rho`` is the common normalized computational grid used by fusdb.  It is not
itself a physical radial convention.  Geometry-dependent normalized coordinates
(e.g. enclosed-volume, minor-radius or flux coordinates) can be tabulated on
that grid and passed explicitly to the averaging helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any = None) -> Any:
    """Trapezoidal integration over the last axis, implemented with NumPy.

    A batched ``(N, P)`` stack keeps the reduced axis as size 1, returning
    ``(N, 1)`` rather than ``(N,)``.  In the popcon batched namespace scalars
    carry a trailing ``1`` axis, so a ``(N, 1)`` scalar times this integral
    stays ``(N, 1)`` instead of broadcasting into an ``(N, N)`` outer product;
    a plain ``(P,)`` profile still integrates to a scalar.
    """
    arr = np.asarray(y, dtype=float)
    d = 1.0 if x is None else np.diff(np.asarray(x, dtype=float))
    return np.sum(d * (arr[..., 1:] + arr[..., :-1]) / 2.0, axis=-1, keepdims=arr.ndim > 1)


def coordinate_average(profile: Any, coordinate: Any) -> Any:
    """Return the normalized average of ``profile`` over ``coordinate``.

    The coordinate is a monotonic one-dimensional mapping tabulated on the
    same last-axis grid as the profile.  Its absolute range is irrelevant: the
    integral is divided by the coordinate span.  This makes normalized
    geometry mappings usable like unit conversions without duplicating the
    physical profile on every coordinate convention.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    coord = np.asarray(coordinate, dtype=float)
    if coord.ndim == 0 or coord.shape[-1] != arr.shape[-1]:
        raise ValueError("profile and coordinate must share the same last-axis grid")
    delta = np.diff(coord, axis=-1)
    if not np.all(np.isfinite(coord)) or np.any(delta <= 0.0):
        raise ValueError("profile coordinate must be finite and strictly increasing")
    denom = trapezoid(np.ones_like(arr, dtype=float), x=coord)
    if np.any(np.asarray(denom, dtype=float) <= 0.0):
        raise ValueError("profile coordinate has zero span")
    return trapezoid(arr, x=coord) / denom


def line_average(profile: Any, rho: Any) -> Any:
    """Return the straight average over the supplied normalized coordinate.

    This is a coordinate average, not by itself a physical diagnostic chord.
    Device-specific line-average relations should pass the coordinate whose
    physical meaning they explicitly document.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    r = np.asarray(rho, dtype=float)
    if r.ndim == 1 and arr.shape[-1] == r.size and r.size > 1:
        width = float(r[-1] - r[0])
        if width > 0.0:
            return trapezoid(arr, x=r) / width
    return np.mean(arr, axis=-1, keepdims=arr.ndim > 1)


def volume_average(profile: Any, rho: Any, *, v_norm: Any | None = None) -> Any:
    """Return the volume average of a profile.

    When ``v_norm`` is supplied it is the geometry-provided normalized enclosed
    volume ``V(<rho) / V_p`` tabulated on the common ``rho`` grid, and the
    average is simply ``integral(profile, d v_norm)`` normalized by its span.

    When ``v_norm`` is omitted the established self-similar ``rho`` weighting
    is retained exactly for backward compatibility.  For the default tokamak
    convention this corresponds to ``v_norm = rho**2`` in the continuum; the
    legacy discrete formula is intentionally preserved until geometry mappings
    are wired through the relation graph, so this refactor introduces no
    numerical regression by itself.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    if v_norm is not None:
        return coordinate_average(arr, v_norm)
    r = np.asarray(rho, dtype=float)
    if r.ndim == 1 and arr.shape[-1] == r.size and r.size > 1:
        denom = float(trapezoid(r, x=r))
        if denom > 0.0:
            return trapezoid(arr * r, x=r) / denom
    return line_average(arr, rho)


def normalized_shape(profile: Any, rho: Any, *, v_norm: Any | None = None) -> tuple[Any, Any]:
    """Return ``(average, shape)`` with unit volume-average shape.

    This is the canonical profile decomposition used by the solver:
    ``profile = average * shape`` and ``volume_average(shape) == 1``.  A zero
    average has no meaningful amplitude-normalized shape, so the uniform shape
    is used as the neutral fallback, matching fusdb's existing profile default.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr, np.asarray(1.0)
    avg = np.asarray(volume_average(arr, rho, v_norm=v_norm), dtype=float)
    if avg.size != 1:
        # Batched callers keep one scalar average per leading batch dimension;
        # broadcasting it over the profile axis is the intended decomposition.
        expanded = avg
    else:
        expanded = float(avg.reshape(-1)[0])
    if np.all(np.abs(avg) <= 1.0e-300):
        return avg, np.ones_like(arr, dtype=float)
    shape = arr / expanded
    shape_avg = np.asarray(volume_average(shape, rho, v_norm=v_norm), dtype=float)
    if np.all(np.abs(shape_avg) > 1.0e-300):
        shape = shape / shape_avg
    return avg, shape
