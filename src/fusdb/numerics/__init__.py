"""Low-level numerical primitives shared by FusDB core modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import re

import numpy as np

# Width of the numerical band treated as "on a domain boundary" for PROFILE
# values: a profile element within ZERO_TOL of a physical bound is projected
# onto the corresponding solver bound (and back), so an edge stored as T = 0
# is evaluated just inside the domain instead of in a singular formula.
#
# It is an absolute magnitude, so it is deliberately NOT used where it would be
# compared against a physical value of unknown scale: domain-violation rows use
# the closed domain, exclusive bounds are checked as written, and scalar
# boundary projection triggers on exact equality.  See TODO, "ZERO_TOL as an
# absolute magnitude" -- a 1e-12 band is meaningless for a variable that lives
# at 1e-30, and treating it as one froze a whole reconcile at its start point.
# It still relaxes an INCLUSIVE bound in value_in_domain, which is genuine
# floating-point-noise absorption rather than a scale comparison.
ZERO_TOL = 1e-12

# Relative width of the band treated as "on an inclusive domain bound" by
# value_in_domain, applied to bounds away from zero (ZERO_TOL is the floor, and
# the whole slack for a bound AT zero).  Dimensionless on purpose -- a ratio
# carries no scale assumption, whereas one absolute band cannot serve both a
# bound at 0 and a bound at 1.  Sized to the least-squares convergence
# tolerance: a solved value sits within ~1e-8 relative of the point the solver
# stopped at, so anything further outside a bound is a real violation.
BOUNDARY_REL_SLACK = 1e-8


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

    ``zero_tol`` only ever RELAXES an inclusive bound, absorbing convergence
    noise just outside it.  It is deliberately not applied to an exclusive
    bound: there it would demand a positive MARGIN rather than absorb noise, and
    since it is one global magnitude for every variable that margin is
    meaningless for anything whose values are much smaller (L_int lives at
    ~1e-30 and was reported as violating ``(0, inf)`` because it could not clear
    1e-12).  An exclusive bound is checked as written -- ``x > lower`` -- which
    is scale-free and still rejects the exact boundary value.

    The inclusive slack is ``max(zero_tol, |bound| * BOUNDARY_REL_SLACK)``: an
    absolute floor for a bound AT zero (where a relative slack would vanish, and
    where hiding a sign error matters -- P_brem_imp must not sit at -0.5 MW
    unnoticed), and a RELATIVE band for a bound away from zero, since a solved
    value only ever lands within the solver's own convergence tolerance of it.
    A single absolute band cannot serve both: ``Z_eff >= 1`` was unusable as a
    domain because a pure-hydrogenic plasma converges to 1 - 4.5e-10 and the
    1e-12 band called that a physics violation.  The ratio is dimensionless, so
    it carries no scale assumption of its own.

    Args:
        value: Scalar or array.
        domain: Parsed domain tuple.
        zero_tol: Absolute floor for the slack allowed outside an inclusive bound.

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
        slack = max(zero_tol, abs(float(lower)) * BOUNDARY_REL_SLACK)
        ok = arr >= lower - slack if lower_inc else arr > lower
        if not bool(np.all(ok)):
            return False
    if upper is not None:
        slack = max(zero_tol, abs(float(upper)) * BOUNDARY_REL_SLACK)
        ok = arr <= upper + slack if upper_inc else arr < upper
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


def signed_scalar_grid(lower: float, upper: float, *, decades: int, step: int, dense: bool = False) -> list[float]:
    """Return signed log-spaced start candidates inside ``[lower, upper]``.

    Candidate points are ``+/- 10**k`` for ``k`` in ``range(-decades,
    decades + 1, step)``, clipped to the interval.  With ``dense=True`` the
    grid additionally includes zero, the finite interval endpoints and a
    21-point linear fill between finite bounds (used by the standalone
    relation inverse solve, which brackets sign changes on this grid).

    Args:
        lower: Interval lower bound (may be ``-inf``).
        upper: Interval upper bound (may be ``+inf``).
        decades: Magnitude range of the log grid.
        step: Exponent stride of the log grid.
        dense: Include zero, endpoints and the linear fill.

    Returns:
        Sorted unique candidate values.
    """
    points: set[float] = set()

    def add(value: float) -> None:
        if np.isfinite(value) and lower <= value <= upper:
            points.add(float(value))

    for exponent in range(-decades, decades + 1, step):
        magnitude = float(10.0**exponent)
        add(magnitude)
        add(-magnitude)
    if dense:
        add(0.0)
        if np.isfinite(lower):
            add(lower)
        if np.isfinite(upper):
            add(upper)
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            for value in np.linspace(lower, upper, 21):
                add(float(value))
    return sorted(points)


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


from ..profiles.numerics import line_average, trapezoid, volume_average
