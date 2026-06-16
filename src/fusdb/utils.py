"""General utilities for FusDB numeric relation solving."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import re

import numpy as np


def unique_preserve_order(items: Iterable[Any]) -> tuple[str, ...]:
    """Return unique string values while preserving first occurrence order.

    Args:
        items: Values convertible to strings.

    Returns:
        Tuple of unique string values.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def normalize_tag(tag: str) -> str:
    """Normalize one tag.

    Args:
        tag: Raw tag text.

    Returns:
        Lowercase stripped tag.
    """
    text = str(tag).strip().lower()
    if not text:
        raise ValueError("Tag cannot be empty.")
    return text


def normalize_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normalize tags while preserving order.

    Args:
        tags: None, one string, or an iterable of strings.

    Returns:
        Tuple of normalized tags.
    """
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (normalize_tag(tags),)
    return unique_preserve_order(normalize_tag(tag) for tag in tags)


def parse_constraint_specs(spec: Any) -> tuple[tuple[str, bool], ...]:
    """Normalize constraint specs to ``(expression, enforce)`` pairs.

    Args:
        spec: None, one string, or iterable of strings / ``[string, bool]``.

    Returns:
        Tuple of normalized constraint specs.
    """
    if spec is None:
        return ()
    if isinstance(spec, str):
        return ((spec, True),)
    if isinstance(spec, Mapping):
        raise TypeError("constraints must be a string or iterable of strings/pairs.")
    out: list[tuple[str, bool]] = []
    for item in spec:
        if isinstance(item, str):
            out.append((item, True))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            out.append((item[0], bool(item[1])))
        else:
            raise TypeError("Constraint entries must be strings or [text, enforce] pairs.")
    return tuple(out)


def parse_domain(text: Any) -> tuple[float | None, float | None, bool, bool]:
    """Parse compact domain syntax such as ``[0, inf)``.

    Args:
        text: Domain string, two-item sequence, or None.

    Returns:
        ``(lower, upper, lower_inclusive, upper_inclusive)``.
    """
    if text is None:
        return None, None, True, True
    if isinstance(text, (tuple, list)) and len(text) == 2:
        return _finite_or_none(text[0]), _finite_or_none(text[1]), True, True
    raw = str(text).strip()
    match = re.fullmatch(r"([\[\(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])", raw)
    if not match:
        raise ValueError(f"Invalid domain {text!r}; expected e.g. '[0, inf)'.")
    left, lo, hi, right = match.groups()
    return _finite_or_none(lo), _finite_or_none(hi), left == "[", right == "]"


def _finite_or_none(value: Any) -> float | None:
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity", "none", "null"}:
        return None
    if text in {"-inf", "-infinity"}:
        return None
    return float(value)


def domain_bounds_for_solver(
    domain: tuple[float | None, float | None, bool, bool],
    *,
    zero_tol: float,
) -> tuple[float | None, float | None]:
    """Convert a parsed domain to closed numerical bounds.

    Args:
        domain: Parsed domain tuple.
        zero_tol: Offset used for open finite bounds.

    Returns:
        Lower and upper bounds, with None for unbounded sides.
    """
    lower, upper, lower_inc, upper_inc = domain
    lb = lower
    ub = upper
    if lb is not None and not lower_inc:
        lb = lb + zero_tol
    if ub is not None and not upper_inc:
        ub = ub - zero_tol
    if lb is not None and ub is not None and lb > ub:
        raise ValueError(f"Empty numerical domain after open-bound offset: {domain!r}.")
    return lb, ub


def scipy_bounds(
    domain: tuple[float | None, float | None, bool, bool],
    *,
    zero_tol: float,
) -> tuple[float, float]:
    """Return SciPy-compatible bounds for a parsed domain.

    Args:
        domain: Parsed domain tuple.
        zero_tol: Offset used for open finite bounds.

    Returns:
        Bounds using infinities for unbounded sides.
    """
    lb, ub = domain_bounds_for_solver(domain, zero_tol=zero_tol)
    return -np.inf if lb is None else float(lb), np.inf if ub is None else float(ub)


def validate_solver_domain(
    name: str,
    domain: tuple[float | None, float | None, bool, bool],
    solver_domain: tuple[float | None, float | None, bool, bool],
) -> None:
    """Validate that a solver domain is inside the physical domain.

    Args:
        name: Variable name used in error messages.
        domain: Physical domain.
        solver_domain: Numerical solver domain.
    """
    d_lo, d_hi, d_lo_inc, d_hi_inc = domain
    s_lo, s_hi, s_lo_inc, s_hi_inc = solver_domain
    if d_lo is not None and s_lo is not None:
        if s_lo < d_lo or (s_lo == d_lo and s_lo_inc and not d_lo_inc):
            raise ValueError(f"Variable {name!r} solver_domain lower bound is outside domain.")
    if d_hi is not None and s_hi is not None:
        if s_hi > d_hi or (s_hi == d_hi and s_hi_inc and not d_hi_inc):
            raise ValueError(f"Variable {name!r} solver_domain upper bound is outside domain.")
    if s_lo is not None and s_hi is not None:
        if s_lo > s_hi or (s_lo == s_hi and not (s_lo_inc and s_hi_inc)):
            raise ValueError(f"Variable {name!r} solver_domain is empty.")


def value_in_domain(value: Any, domain: tuple[float | None, float | None, bool, bool], *, zero_tol: float = 0.0) -> bool:
    """Return whether all numeric values are inside a domain.

    Args:
        value: Scalar or array.
        domain: Parsed domain tuple.
        zero_tol: Boundary tolerance.

    Returns:
        True when every finite value is inside the domain.
    """
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    lower, upper, lower_inc, upper_inc = domain
    if lower is not None:
        ok = arr >= lower - zero_tol if lower_inc else arr > lower + zero_tol
        if not bool(np.all(ok)):
            return False
    if upper is not None:
        ok = arr <= upper + zero_tol if upper_inc else arr < upper - zero_tol
        if not bool(np.all(ok)):
            return False
    return True


def coerce_numeric_value(value: Any) -> Any:
    """Convert numeric-looking YAML values to Python/NumPy numeric values.

    Args:
        value: Raw user value.

    Returns:
        Numeric value where possible, otherwise the original value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray([coerce_numeric_value(item) for item in value], dtype=float)
        except Exception:
            return value
    return value


def coerce_to_shape(
    name: str,
    value: Any,
    *,
    is_profile: bool,
    size: int | None,
    squeeze_scalar: bool = False,
    reject_nan: bool = False,
) -> tuple[Any, int | None]:
    """Coerce a numeric value to a scalar or 1-D profile.

    Shared by ``Variable._normalize_value`` (public/user values) and
    ``RelationSystem._coerce_to_registry_shape`` (solver namespaces); the
    flags capture the few rule differences between those callers.

    Args:
        name: Variable name used in error messages.
        value: Scalar or array-like numeric value.
        is_profile: Whether the variable is a 1-D profile (registry shape 1).
        size: Known profile length, or None to infer it from a 1-D value.
        squeeze_scalar: Accept a single-element array as a scalar value.
        reject_nan: Raise when any element is NaN.

    Returns:
        ``(coerced_value, size)`` with the possibly inferred profile size.
    """
    arr = np.asarray(value, dtype=float)
    if reject_nan and np.any(np.isnan(arr)):
        raise ValueError(f"Variable {name!r} contains nan.")
    if not is_profile:
        if arr.ndim == 0:
            return float(arr), size
        flat = arr.reshape(-1)
        if squeeze_scalar and flat.size == 1:
            return float(flat[0]), size
        raise ValueError(f"Scalar variable {name!r} received non-scalar value with shape {arr.shape}.")
    if arr.ndim == 0:
        if size is None:
            return float(arr), None
        return np.full(int(size), float(arr), dtype=float), size
    if arr.ndim == 1:
        if size is None:
            return arr.astype(float), int(arr.shape[0])
        if int(size) != int(arr.shape[0]):
            raise ValueError(f"Variable {name!r} size mismatch: {size} vs {arr.shape[0]}.")
        return arr.astype(float), size
    raise ValueError(f"Profile variable {name!r} value must be scalar or 1D.")


def compare_numeric(
    lhs: Any,
    op: str,
    rhs: Any,
    *,
    scale: Any,
    rel_tol: float,
    abs_tol: float = 0.0,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Evaluate an equality or inequality using tolerance-width residuals.

    ``scale`` is the physical/current/reference magnitude used for relative
    tolerance.  It is not itself the residual denominator.  The residual
    denominator is the actual tolerance width

        max(abs_tol, rel_tol * scale)

    so an error of one residual unit means one allowed tolerance width.  Bounds
    and unbounded domains must not be passed as ``scale``.
    """
    left = np.asarray(lhs, dtype=float)
    right = np.asarray(rhs, dtype=float)
    scl = np.maximum(np.asarray(scale, dtype=float), 1.0e-300)
    tol_width = np.maximum(float(abs_tol), float(rel_tol) * scl)
    tol_width = np.maximum(tol_width, 1.0e-300)
    diff = left - right
    if op == "==":
        violation = np.abs(diff)
        residual = diff / tol_width
    elif op in {"<=", "<"}:
        violation = np.maximum(diff, 0.0)
        residual = violation / tol_width
    elif op in {">=", ">"}:
        violation = np.maximum(-diff, 0.0)
        residual = violation / tol_width
    else:
        raise ValueError(f"Unsupported comparison operator {op!r}.")
    ok = bool(np.all(violation <= tol_width))
    return ok, np.asarray(residual, dtype=float).reshape(-1), np.asarray(violation, dtype=float).reshape(-1)


def safe_max_abs(value: Any, default: float = 0.0) -> float:
    """Return max absolute finite magnitude or a default.

    Args:
        value: Scalar or array.
        default: Fallback value.

    Returns:
        Non-negative finite magnitude.
    """
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.max(np.abs(finite)))
