"""Shared helpers for relation modules."""

from typing import Any

import numpy as np
from fusdb.utils import trapezoid


def _positive_denominator(value: Any, *, name: str) -> Any:
    """Return a finite positive denominator for nonlinear least-squares.

    Relation functions should not abort during intermediate SciPy iterations.
    Domain/bounds and final residual checks decide whether the final state is
    acceptable.  This helper therefore clips invalid or non-positive temporary
    denominators to a tiny positive value instead of raising.
    """
    arr = np.asarray(value, dtype=float)
    arr = np.nan_to_num(arr, nan=1e-300, posinf=1e300, neginf=1e-300)
    arr = np.maximum(arr, 1e-300)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _species_fraction(numerator: Any, denominator: Any, *, name: str) -> Any:
    """Return the integrated species fraction from density profiles.

    The ratio uses grid-integrated densities, so a profile whose edge value is
    exactly zero does not create an indeterminate pointwise ``0/0`` sample.
    For shape-proportional profiles this equals the pointwise fraction.
    """
    num = np.asarray(numerator, dtype=float).reshape(-1)
    den = np.asarray(denominator, dtype=float).reshape(-1)
    species = float(trapezoid(num)) if num.size > 1 else float(num[0])
    total = float(trapezoid(den)) if den.size > 1 else float(den[0])
    total = float(_positive_denominator(total, name=f"{name} denominator"))
    return species / total
