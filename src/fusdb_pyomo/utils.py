"""General utilities for FusDB numeric relation solving."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import math
import re

import numpy as np


def unique_preserve_order(items: Iterable[Any]) -> tuple[str, ...]:
    """Return unique string values while preserving first occurrence order.

    Args:
        items: Values convertible to strings.

    Returns:
        Tuple of unique strings.
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
    """Normalize one tag by stripping whitespace and lowercasing it.

    Args:
        tag: Tag text.

    Returns:
        Normalized tag.
    """
    text = str(tag).strip().lower()
    if not text:
        raise ValueError("Tag cannot be empty.")
    return text


def normalize_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Normalize tags while preserving first occurrence order.

    Args:
        tags: None, one tag, or an iterable of tags.

    Returns:
        Tuple of normalized tags.
    """
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (normalize_tag(tags),)
    return unique_preserve_order(normalize_tag(tag) for tag in tags)


def parse_constraint_specs(spec: Any) -> tuple[tuple[str, bool], ...]:
    """Normalize constraint definitions to ``(text, enforce)`` pairs.

    Args:
        spec: None, one string, or an iterable of strings or ``[text, enforce]`` pairs.

    Returns:
        Tuple of normalized constraint records.
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
            continue
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], bool)
        ):
            out.append((item[0], item[1]))
            continue
        raise TypeError("Constraint entries must be strings or [text, enforce] pairs.")
    return tuple(out)


def parse_domain(text: Any) -> tuple[float | None, float | None, bool, bool]:
    """Parse compact domain syntax such as ``"[0, inf)"``.

    Args:
        text: Domain string, list/tuple pair, or None.

    Returns:
        ``(lower, upper, lower_inclusive, upper_inclusive)``. Infinite bounds are
        returned as ``None``.
    """
    if text is None:
        return None, None, True, True
    if isinstance(text, (tuple, list)) and len(text) == 2:
        lo, hi = text
        return _finite_or_none(lo), _finite_or_none(hi), True, True

    raw = str(text).strip()
    match = re.fullmatch(r"([\[\(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\]\)])", raw)
    if not match:
        raise ValueError(f"Invalid domain {text!r}; expected e.g. '[0, inf)'.")
    left_bracket, lo, hi, right_bracket = match.groups()
    return (
        _finite_or_none(lo),
        _finite_or_none(hi),
        left_bracket == "[",
        right_bracket == "]",
    )


def _finite_or_none(value: Any) -> float | None:
    """Convert finite domain bound text to float and infinities to None."""
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
    """Return closed numerical solver bounds for a parsed domain.

    Args:
        domain: Parsed domain tuple.
        zero_tol: Offset used to approximate open finite bounds.

    Returns:
        Lower and upper bounds. ``None`` means unbounded on that side.
    """
    lower, upper, lower_inclusive, upper_inclusive = domain
    lb = lower
    ub = upper
    if lb is not None and not lower_inclusive:
        lb = lb + zero_tol
    if ub is not None and not upper_inclusive:
        ub = ub - zero_tol
    if lb is not None and ub is not None and lb > ub:
        raise ValueError(f"Solver bounds are empty after open-bound offset: {domain!r}.")
    return lb, ub


def scipy_bounds(
    domain: tuple[float | None, float | None, bool, bool],
    *,
    zero_tol: float,
) -> tuple[float, float]:
    """Return SciPy-compatible finite-or-infinite bounds.

    Args:
        domain: Parsed domain tuple.
        zero_tol: Offset used to approximate open finite bounds.

    Returns:
        Lower and upper bounds using ``-/+np.inf`` for unbounded sides.
    """
    lb, ub = domain_bounds_for_solver(domain, zero_tol=zero_tol)
    return (-np.inf if lb is None else float(lb), np.inf if ub is None else float(ub))


def value_in_domain(value: Any, domain: tuple[float | None, float | None, bool, bool], *, zero_tol: float = 0.0) -> bool:
    """Return whether a scalar or array is inside a registry domain.

    Args:
        value: Numeric value or array.
        domain: Parsed domain tuple.
        zero_tol: Tolerance for numerical boundary comparisons.

    Returns:
        True if every value is inside the domain.
    """
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    lower, upper, lower_inclusive, upper_inclusive = domain
    if lower is not None:
        ok = arr >= lower - zero_tol if lower_inclusive else arr > lower + zero_tol
        if not bool(np.all(ok)):
            return False
    if upper is not None:
        ok = arr <= upper + zero_tol if upper_inclusive else arr < upper - zero_tol
        if not bool(np.all(ok)):
            return False
    return True


def validate_solver_domain(
    name: str,
    domain: tuple[float | None, float | None, bool, bool],
    solver_domain: tuple[float | None, float | None, bool, bool],
) -> None:
    """Validate that the numerical solver domain is inside the physical domain.

    Args:
        name: Variable name used in error messages.
        domain: Physical/API domain.
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


def coerce_numeric_value(value: Any) -> Any:
    """Convert YAML numeric-looking values to floats or NumPy arrays.

    Args:
        value: Raw value from YAML or user input.

    Returns:
        Numeric value when possible; otherwise the original value.
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


def finite_array(value: Any, *, name: str) -> np.ndarray:
    """Convert a value to a finite float array.

    Args:
        value: Numeric value.
        name: Label used in error messages.

    Returns:
        NumPy float array.
    """
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} is not numeric.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nan or inf.")
    return arr


def max_abs(value: Any) -> float:
    """Return max absolute finite magnitude, or zero for empty values."""
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def relative_error(lhs: Any, rhs: Any) -> float | None:
    """Compute a max relative residual for scalars or arrays.

    Args:
        lhs: First value.
        rhs: Second value.

    Returns:
        Max pointwise relative error, or None if inputs are not finite numeric.
    """
    try:
        left = finite_array(lhs, name="lhs")
        right = finite_array(rhs, name="rhs")
        raw = np.abs(left - right)
        denom = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
        return float(np.max(raw / denom))
    except Exception:
        return None


def compare_numeric(lhs: Any, op: str, rhs: Any, *, rel_tol: float, zero_tol: float) -> bool | None:
    """Check a numeric comparison.

    Args:
        lhs: Left value.
        op: Comparison operator.
        rhs: Right value.
        rel_tol: Relative tolerance for equality.
        zero_tol: Absolute threshold for strict comparisons.

    Returns:
        True/False when evaluable, otherwise None.
    """
    try:
        left = finite_array(lhs, name="lhs")
        right = finite_array(rhs, name="rhs")
        if op == "==":
            err = relative_error(left, right)
            return bool(err is not None and err <= rel_tol)
        if op in {"<", "<="}:
            margin = zero_tol if op == "<" else -zero_tol
            return bool(np.all(left <= right - margin))
        if op in {">", ">="}:
            margin = zero_tol if op == ">" else -zero_tol
            return bool(np.all(left >= right + margin))
    except Exception:
        return None
    raise ValueError(f"Unsupported operator {op!r}.")


def safe_float(value: Any) -> float | None:
    """Return value as float when conversion is safe, otherwise None."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def is_symbolic(value: Any) -> bool:
    """Return False for the SciPy-only backend.

    Args:
        value: Any value.

    Returns:
        Always False. Relation functions are evaluated numerically only.
    """
    return False
