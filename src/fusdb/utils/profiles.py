"""Numerical integration and averaging helpers for radial profiles."""

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


def line_average(profile: Any, rho: Any) -> Any:
    """Return the radial line average of a profile tabulated over ``rho``."""
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


def volume_average(profile: Any, rho: Any) -> Any:
    """Return a flux-volume average using the self-similar ``rho`` weighting."""
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return np.asarray(0.0)
    r = np.asarray(rho, dtype=float)
    if r.ndim == 1 and arr.shape[-1] == r.size and r.size > 1:
        denom = float(trapezoid(r, x=r))
        if denom > 0.0:
            return trapezoid(arr * r, x=r) / denom
    return line_average(arr, rho)
