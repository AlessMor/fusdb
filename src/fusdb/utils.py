"""Shared numeric and utility helpers for the fusdb package.

This module provides common utility functions for:
- Numeric comparisons with tolerances
- Tag normalization and comparison
- Country name normalization
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pycountry
import numpy as np


def within_tolerance(
    a: object,
    b: object,
    *,
    rel_tol: float = 0.0,
) -> bool:
    """Check if two values are equal within a relative tolerance.

    The comparison is: ``|a - b| <= rel_tol * scale``
    where scale is ``max(abs(a), abs(b), 1.0)``.
    
    Args:
        a (object): First value (must be convertible to float).
        b (object): Second value (must be convertible to float).
        rel_tol (float): Relative tolerance (fraction of larger value).
    Returns:
        bool: True if values are within tolerance, False otherwise.
              Returns False if either value is None, non-numeric, or infinite/NaN.
        
    Example:
        >>> within_tolerance(1.0, 1.001, rel_tol=0.01)
        True
    """
    # Can't compare None or non-numeric values
    if a is None or b is None:
        return False
    
    # Try to convert to floats
    try:
        av = float(a)
        bv = float(b)
    except Exception:
        return False
    
    # Reject infinite or NaN values
    if not math.isfinite(av) or not math.isfinite(bv):
        return False
    
    # Use relative tolerance scaled by larger value
    scale = max(abs(av), abs(bv), 1.0)
    return abs(av - bv) <= rel_tol * scale


def all_tolerances(
    left: object,
    right: object,
    *,
    rel_tol: float,
) -> bool:
    """Return whether array-like values satisfy relative tolerance checks.

    Args:
        left: First scalar or array-like value.
        right: Second scalar or array-like value.
        rel_tol: Relative tolerance used elementwise.

    Returns:
        ``True`` when shapes match and every element is within tolerance.
    """
    # Convert both payloads through NumPy so scalars and arrays share one path.
    try:
        left_arr = np.asarray(left, dtype=float)
        right_arr = np.asarray(right, dtype=float)
    except Exception:
        return False

    # Shape mismatch means the values cannot represent the same runtime state.
    if left_arr.shape != right_arr.shape:
        return False

    # Check exact equality when no relative tolerance is available.
    diff = np.abs(left_arr - right_arr)
    if rel_tol <= 0.0:
        return bool(np.all(diff == 0.0))

    # Scale relative tolerance elementwise by the larger local magnitude.
    scale = np.maximum(np.maximum(np.abs(left_arr), np.abs(right_arr)), 1.0)
    return bool(np.all(diff <= rel_tol * scale))


def relative_change(current: float, target: float) -> float:
    """Return a movement score that penalizes order-of-magnitude drifts.

    Args:
        current: Baseline scalar value.
        target: Candidate scalar value.

    Returns:
        Dimensionless movement score for ranking/guarding candidate updates.
    """
    # Compute linear and symmetric relative movement terms.
    linear = abs(target - current) / max(abs(current), 1.0)
    denom = abs(current) + abs(target)
    symmetric = 0.0 if denom == 0.0 else (2.0 * abs(target - current) / denom)

    # Add an order-of-magnitude term when values keep the same sign.
    order = 0.0
    if current != 0.0 and target != 0.0 and (current * target) > 0.0:
        try:
            order = abs(math.log10(abs(target)) - math.log10(abs(current)))
        except Exception:
            order = 0.0

    # Return the strictest movement signal among the three.
    return max(linear, symmetric, order)


def brent_root(
    f,
    a: float,
    b: float,
    fa: float,
    fb: float,
    *,
    rel_tol: float,
    max_iter: int,
) -> float | None:
    """Return root in ``[a, b]`` using a Brent-style method.

    Args:
        f: Callable returning residual values.
        a: Lower bracket point.
        b: Upper bracket point.
        fa: Residual at ``a``.
        fb: Residual at ``b``.
        rel_tol: Relative convergence tolerance.
        max_iter: Maximum iteration count.

    Returns:
        Root value, or ``None`` when the bracket/iteration is invalid.
    """
    # Validate finite bracket endpoints and sign change.
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(fa) and math.isfinite(fb)):
        return None
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return None

    # Track the current bracket and interpolation step history.
    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = max(rel_tol * max(abs(b), 1.0), 1e-12)
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b

        # Prefer inverse interpolation when stable, otherwise bisect.
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if q != 0.0 and 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        # Advance the current estimate by the chosen interpolation/bisection step.
        a, fa = b, fb
        if abs(d) > tol:
            b += d
        else:
            b += tol if m > 0.0 else -tol

        fb = f(b)
        if fb is None or not math.isfinite(fb):
            return None

    return None


def safe_float(value: object) -> float | None:
    """Convert value to float, returning None if conversion fails or non-finite.
    
    Args:
        value (object): Value to convert.
        
    Returns:
        float | None: Float conversion result, or None if conversion fails or value is infinite/NaN.
    """
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def as_profile_array(value: object) -> np.ndarray | None:
    """Return value as a valid 1D finite profile array.

    Args:
        value: Candidate profile payload.

    Returns:
        A 1D finite float NumPy array, or None when invalid.
    """
    try:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).all():
            return arr
    except Exception:
        pass
    return None


def mean_profile(value: object) -> float | None:
    """Return the arithmetic mean of a profile payload.

    Args:
        value: Candidate scalar/array payload.

    Returns:
        Profile mean when ``value`` is a finite 1D array, else ``None``.
    """
    # Parse profile payload and return None when payload is not profile-like.
    arr = as_profile_array(value)
    if arr is None:
        return None
    # Compute one explicit mean scalar used by scalar-first paths.
    return float(np.mean(arr))


def scalarize_value(value: object) -> object | None:
    """Convert one value to scalar semantics for scalar-first solving.

    Args:
        value: Candidate runtime payload.

    Returns:
        Profile mean for profiles, finite float for scalar-like payloads,
        original value when no safe scalar conversion exists, or ``None``.
    """
    # Reduce profile payloads explicitly by their mean.
    profile_mean = mean_profile(value)
    if profile_mean is not None:
        return profile_mean

    # Convert scalar-like payloads to finite float when possible.
    scalar = safe_float(value)
    if scalar is not None:
        return scalar

    # Keep non-scalar payloads untouched to avoid silent coercion.
    return value


def scalarize_mapping(
    values: Mapping[str, object],
    *,
    ndim_lookup: Callable[[str], int],
) -> dict[str, object]:
    """Return a mapping where profile payloads are explicitly scalarized.

    Args:
        values: Input mapping of variable names to runtime payloads.
        ndim_lookup: Callable returning variable dimensionality by name.

    Returns:
        Dictionary with profile variables reduced to means and scalars preserved.
    """
    out: dict[str, object] = {}

    # Walk every value and apply explicit ndim-aware reduction.
    for name, value in values.items():
        if ndim_lookup(name) == 1:
            out[name] = scalarize_value(value)
            continue
        out[name] = value

    return out


def integrate_profile_over_volume(
    profile: object,
    V_p: object,
    *,
    rho: object | None = None,
) -> float | None:
    """Integrate a profile over volume using cfspopcon-style ``d(V/V_p)=2*rho drho``.

    Args:
        profile: Local quantity profile (or scalar value).
        V_p: Total plasma volume.
        rho: Optional normalized radial grid in ``[0, 1]``.

    Returns:
        Integrated scalar value or ``None`` when inputs are not usable.
    """
    V_scalar = safe_float(V_p)
    if V_scalar is None or V_scalar < 0.0:
        return None

    arr = as_profile_array(profile)
    if arr is None:
        scalar = safe_float(profile)
        return None if scalar is None else float(scalar * V_scalar)

    if rho is None:
        rho_arr = np.linspace(0.0, 1.0, arr.size, dtype=float)
    else:
        rho_arr = as_profile_array(rho)
        if rho_arr is None:
            return None
        if rho_arr.size != arr.size:
            x_old = np.linspace(0.0, 1.0, arr.size, dtype=float)
            x_new = np.linspace(0.0, 1.0, rho_arr.size, dtype=float)
            arr = np.interp(x_new, x_old, arr)

    if rho_arr.size > 1 and np.any(np.diff(rho_arr) < 0):
        order = np.argsort(rho_arr)
        rho_arr = rho_arr[order]
        arr = arr[order]

    return float(V_scalar * np.trapezoid(arr * 2.0 * rho_arr, rho_arr))


def integrate_profile(
    profile: object,
    V_p: object = 1.0,
    *,
    error_label: str = "profile",
) -> object:
    """Return a volume-integrated profile with symbolic fallback.

    Args:
        profile: Local quantity profile, scalar value, or symbolic expression.
        V_p: Total plasma volume used for the integration. Defaults to unit volume.
        error_label: Quantity label used in the integration error message.

    Returns:
        Volume-integrated scalar result, or a symbolic expression when symbolic
        placeholders are passed in.

    Raises:
        ValueError: If numeric inputs cannot be integrated.
    """
    # Keep symbolic model builds algebraic instead of forcing numeric integration.
    if getattr(profile, "free_symbols", None) is not None or getattr(V_p, "free_symbols", None) is not None:
        return profile if safe_float(V_p) == 1.0 else profile * V_p

    # Delegate numeric scalars and 1D profiles to the shared volume integrator.
    total = integrate_profile_over_volume(profile, V_p)
    if total is None:
        raise ValueError(f"Cannot integrate {error_label} profile over volume.")
    return total


def normalize_tag(tag: str | None) -> str:
    """Normalize a single tag for consistent comparison.
    
    Converts tag to lowercase and removes non-alphanumeric characters.
    
    Args:
        tag (str | None): Single tag string or None.
        
    Returns:
        str: Normalized tag string (empty string if None/empty).
        
    Examples:
        >>> normalize_tag("Tokamak-Spherical")
        'tokamakspherical'
        >>> normalize_tag(None)
        ''
    """
    if not tag:
        return ""
    return "".join(ch for ch in str(tag).lower() if ch.isalnum())


def normalize_tags_to_tuple(tags: object) -> tuple[str, ...]:
    """Normalize tags and convert to tuple format.
    
    Handles single strings, iterables, or None. Normalizes each tag
    (lowercase, alphanumeric only) for consistent comparison.
    
    Args:
        tags (object): Tags as string, iterable of strings, or None.
        
    Returns:
        tuple[str, ...]: Tuple of normalized tag strings.
        
    Examples:
        >>> normalize_tags_to_tuple("Tokamak-Spherical")
        ('tokamakspherical',)
        >>> normalize_tags_to_tuple(["Tokamak", "H-mode"])
        ('tokamak', 'hmode')
        >>> normalize_tags_to_tuple(None)
        ()
    """
    if tags is None:
        return ()
    
    # Convert single string to list, otherwise iterate through tags
    tag_list = [tags] if isinstance(tags, str) else tags
    
    # Normalize each tag and filter out empty results
    return tuple(norm for tag in tag_list if (norm := normalize_tag(tag)))


def normalize_country(country: str | None) -> str | None:
    """Normalize a country name to its official name.
    
    Uses the pycountry library to convert country codes, aliases, or
    informal names to official country names.
    
    Args:
        country (str | None): Country name, code (ISO 3166), or alias.
    
    Returns:
        str | None: Official country name from pycountry, or the original string
                    if no match found. Returns None if input is None/empty.
        
    Example:
        >>> normalize_country("USA")
        'United States'
        >>> normalize_country("UK")
        'United Kingdom'
        >>> normalize_country("Unknown")
        'Unknown'
    """
    if not country:
        return None
    
    # Attempt to look up and normalize the country name
    try:
        match = pycountry.countries.lookup(country)
        return match.name
    except Exception:
        # Not found - return original string
        return country
