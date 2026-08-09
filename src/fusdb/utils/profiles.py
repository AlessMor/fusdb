"""Numerical integration and averaging helpers for one-dimensional profiles.

``rho`` is the common normalized computational grid used by fusdb. It is not
itself a physical radial convention. Geometry-dependent normalized coordinates
and integration weights are tabulated on that grid and passed explicitly when
needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def trapezoid(y: Any, x: Any = None) -> Any:
    """Trapezoidal integration over the last axis, implemented with NumPy.

    A batched ``(N, P)`` stack keeps the reduced axis as size 1, returning
    ``(N, 1)`` rather than ``(N,)``. In the popcon batched namespace scalars
    carry a trailing ``1`` axis, so a ``(N, 1)`` scalar times this integral
    stays ``(N, 1)`` instead of broadcasting into an ``(N, N)`` outer product;
    a plain ``(P,)`` profile still integrates to a scalar.
    """
    arr = np.asarray(y, dtype=float)
    d = 1.0 if x is None else np.diff(np.asarray(x, dtype=float))
    return np.sum(d * (arr[..., 1:] + arr[..., :-1]) / 2.0, axis=-1, keepdims=arr.ndim > 1)


def coordinate_average(profile: Any, coordinate: Any) -> Any:
    """Return the normalized average of ``profile`` over ``coordinate``.

    ``coordinate`` is a finite, strictly increasing normalized coordinate
    mapping tabulated on the same last-axis grid as ``profile``. Its absolute
    range is irrelevant because the integral is divided by the coordinate span.
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
    Device-specific line-average relations must pass the coordinate whose
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


def weighted_average(profile: Any, rho: Any, weight: Any) -> Any:
    """Return ``integral(profile * weight d rho) / integral(weight d rho)``.

    Geometry can therefore provide an integration measure such as a quantity
    proportional to ``dV / d rho`` without changing the common computational
    grid or duplicating the physical profile in another coordinate system.
    The weight may vanish at isolated points (for example at the magnetic axis)
    but must be finite, non-negative and have a positive integral.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    r = np.asarray(rho, dtype=float)
    w = np.asarray(weight, dtype=float)
    if r.ndim != 1 or r.size != arr.shape[-1] or w.ndim == 0 or w.shape[-1] != arr.shape[-1]:
        raise ValueError("profile, rho and weight must share the same last-axis grid")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("profile integration weight must be finite and non-negative")
    denom = trapezoid(w, x=r)
    if np.any(np.asarray(denom, dtype=float) <= 0.0):
        raise ValueError("profile integration weight has zero integral")
    return trapezoid(arr * w, x=r) / denom


def volume_average(
    profile: Any,
    rho: Any,
    *,
    weight: Any | None = None,
    v_norm: Any | None = None,
) -> Any:
    """Return the volume average of a profile.

    Preferred geometry-aware form: ``weight`` is proportional to ``dV/d rho``.
    Its normalization is irrelevant. For the legacy self-similar tokamak
    convention, ``weight=rho`` reproduces fusdb's historical discrete formula
    exactly.

    ``v_norm`` is accepted when geometry naturally supplies normalized enclosed
    volume ``V(<rho)/V_p`` rather than its derivative; the profile is integrated
    directly over that monotonic coordinate.

    Supplying neither retains the historical self-similar ``rho`` weighting for
    backwards compatibility while relations are migrated to explicit geometry.
    """
    if weight is not None and v_norm is not None:
        raise ValueError("volume_average accepts either weight or v_norm, not both")
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    if weight is not None:
        return weighted_average(arr, rho, weight)
    if v_norm is not None:
        return coordinate_average(arr, v_norm)
    r = np.asarray(rho, dtype=float)
    if r.ndim == 1 and arr.shape[-1] == r.size and r.size > 1:
        denom = float(trapezoid(r, x=r))
        if denom > 0.0:
            return trapezoid(arr * r, x=r) / denom
    return line_average(arr, rho)


def normalized_shape(
    profile: Any,
    rho: Any,
    *,
    weight: Any | None = None,
    v_norm: Any | None = None,
) -> tuple[Any, Any]:
    """Return ``(average, shape)`` with unit volume-average shape.

    This is the canonical profile decomposition used by the solver:
    ``profile = average * shape`` and ``volume_average(shape) == 1``. A zero
    average has no amplitude-normalized shape, so the uniform shape is the
    neutral fallback.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr, np.asarray(1.0)
    avg = np.asarray(volume_average(arr, rho, weight=weight, v_norm=v_norm), dtype=float)
    expanded: Any = avg if avg.size != 1 else float(avg.reshape(-1)[0])
    if np.all(np.abs(avg) <= 1.0e-300):
        return avg, np.ones_like(arr, dtype=float)
    shape = arr / expanded
    shape_avg = np.asarray(volume_average(shape, rho, weight=weight, v_norm=v_norm), dtype=float)
    if np.all(np.abs(shape_avg) > 1.0e-300):
        shape = shape / shape_avg
    return avg, shape


def reinterpolate_profile(
    source_profile: Any,
    source_coordinate: Any,
    target_coordinate: Any,
    *,
    endpoint_tol: float = 1.0e-12,
) -> np.ndarray:
    """Interpolate immutable source-profile data onto a current coordinate map.

    This is the numerical primitive used by geometry-dependent profile
    conversion. ``source_coordinate`` is the coordinate attached to the input
    samples; ``target_coordinate`` is the current mapping evaluated on fusdb's
    common ``rho`` grid. The source data are not changed when geometry moves.

    Interpolation is strict: both coordinates must be finite and strictly
    increasing, and the current target range must lie inside the supplied source
    range. Only roundoff-sized endpoint excursions (``endpoint_tol``) are
    clamped. Material extrapolation is rejected rather than hidden.

    A one-dimensional source profile may be evaluated on a one-dimensional or
    batched target mapping. A batched source profile is also supported when its
    leading dimensions exactly match the target mapping's leading dimensions.
    """
    values = np.asarray(source_profile, dtype=float)
    source = np.asarray(source_coordinate, dtype=float)
    target = np.asarray(target_coordinate, dtype=float)
    tol = float(endpoint_tol)
    if tol < 0.0 or not np.isfinite(tol):
        raise ValueError("endpoint_tol must be finite and non-negative")
    if source.ndim != 1 or source.size < 2:
        raise ValueError("source coordinate must be a one-dimensional grid with at least two points")
    if values.ndim == 0 or values.shape[-1] != source.size:
        raise ValueError("source profile and source coordinate must share the same last-axis grid")
    if target.ndim == 0 or target.shape[-1] < 2:
        raise ValueError("target coordinate must have at least two points on its last axis")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("profile interpolation inputs must be finite")
    if np.any(np.diff(source) <= 0.0):
        raise ValueError("source coordinate must be strictly increasing")
    if np.any(np.diff(target, axis=-1) <= 0.0):
        raise ValueError("target coordinate must be strictly increasing")

    lo = float(source[0])
    hi = float(source[-1])
    if np.any(target < lo - tol) or np.any(target > hi + tol):
        target_lo = float(np.min(target))
        target_hi = float(np.max(target))
        raise ValueError(
            f"target coordinate range [{target_lo:g}, {target_hi:g}] is outside "
            f"source coverage [{lo:g}, {hi:g}]"
        )
    clipped = np.clip(target, lo, hi)

    if values.ndim == 1:
        return np.interp(clipped.reshape(-1), source, values).reshape(clipped.shape)

    if values.shape[:-1] != clipped.shape[:-1]:
        raise ValueError(
            "batched source profile and target coordinate must have identical leading dimensions"
        )
    out = np.empty_like(clipped, dtype=float)
    flat_values = values.reshape(-1, values.shape[-1])
    flat_target = clipped.reshape(-1, clipped.shape[-1])
    flat_out = out.reshape(-1, out.shape[-1])
    for index in range(flat_values.shape[0]):
        flat_out[index] = np.interp(flat_target[index], source, flat_values[index])
    return out
