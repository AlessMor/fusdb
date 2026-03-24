"""Shared numeric and utility helpers for the fusdb package.

This module provides common utility functions for:
- Numeric comparisons with tolerances
- Tag normalization and comparison
- Country name normalization
"""
from __future__ import annotations

import math
import warnings
import pycountry
import numpy as np


def within_tolerance(
    a: object,
    b: object,
    *,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> bool:
    """Check if two values are equal within specified tolerances.
    
    Compares two values using both absolute and relative tolerance.
    The comparison is: ``|a - b| <= max(abs_tol, rel_tol * scale)``
    where scale is max(abs(a), abs(b), 1.0).
    
    Args:
        a (object): First value (must be convertible to float).
        b (object): Second value (must be convertible to float).
        rel_tol (float): Relative tolerance (fraction of larger value).
        abs_tol (float): Absolute tolerance (in same units as values).
    
    Returns:
        bool: True if values are within tolerance, False otherwise.
              Returns False if either value is None, non-numeric, or infinite/NaN.
        
    Example:
        >>> within_tolerance(1.0, 1.001, rel_tol=0.01)
        True
        >>> within_tolerance(1.0, 1.001, abs_tol=0.0001)
        False
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
    return abs(av - bv) <= max(abs_tol, rel_tol * scale)


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

def _trapz(y: object, x: object) -> float:
    """Integrate using numpy trapezoid API with backward-compatible fallback."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


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

    return float(V_scalar * _trapz(arr * 2.0 * rho_arr, rho_arr))


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


def compare_plasma_volume_with_integrated_dv(
    *,
    V_p: object,
    rho: object | None = None,
    dV_drho: object | None = None,
    R: object | None = None,
    a: object | None = None,
    kappa: object | None = None,
    rel_tol: float = 0.01,
    abs_tol: float = 0.0,
    warn: bool = False,
) -> tuple[bool, float | None, float | None]:
    """Compare ``V_p`` with a volume reconstructed from ``dV`` integration.

    Priority for reconstructed volume:
    1. ``∫ dV_drho drho`` if ``dV_drho`` is provided.
    2. ``2*pi^2*R*kappa*a^2`` if ``R``, ``a``, and ``kappa`` are provided.
    3. ``V_p*∫2*rho drho`` (cfspopcon Jacobian sanity check).

    Args:
        V_p: Reference plasma volume.
        rho: Optional normalized radial grid.
        dV_drho: Optional Jacobian profile ``dV/drho``.
        R: Major radius for geometric estimate.
        a: Minor radius for geometric estimate.
        kappa: Elongation for geometric estimate.
        rel_tol: Relative tolerance for consistency.
        abs_tol: Absolute tolerance for consistency.
        warn: Emit ``UserWarning`` if mismatch exceeds tolerance.

    Returns:
        ``(ok, integrated_volume, reference_volume)``.
    """
    V_ref = safe_float(V_p)
    if V_ref is None:
        return True, None, None

    V_int: float | None = None
    rho_arr = as_profile_array(rho) if rho is not None else None
    jac_arr = as_profile_array(dV_drho) if dV_drho is not None else None
    if jac_arr is not None:
        if rho_arr is None:
            rho_arr = np.linspace(0.0, 1.0, jac_arr.size, dtype=float)
        elif rho_arr.size != jac_arr.size:
            x_old = np.linspace(0.0, 1.0, jac_arr.size, dtype=float)
            x_new = np.linspace(0.0, 1.0, rho_arr.size, dtype=float)
            jac_arr = np.interp(x_new, x_old, jac_arr)
        V_int = _trapz(jac_arr, rho_arr)
    else:
        Rv = safe_float(R)
        av = safe_float(a)
        kv = safe_float(kappa)
        if Rv is not None and av is not None and kv is not None:
            V_int = float(2.0 * math.pi ** 2 * Rv * kv * av ** 2)
        else:
            if rho_arr is None:
                rho_arr = np.linspace(0.0, 1.0, 101, dtype=float)
            V_int = float(V_ref * _trapz(2.0 * rho_arr, rho_arr))

    ok = within_tolerance(V_int, V_ref, rel_tol=rel_tol, abs_tol=abs_tol)
    if warn and not ok:
        delta = abs(V_int - V_ref) / max(abs(V_ref), 1.0)
        warnings.warn(
            (
                "Plasma volume mismatch: V_p="
                f"{V_ref:.6g}, integral(dV)={V_int:.6g}, rel_delta={delta:.3%} "
                f"(tol={rel_tol:.3%})."
            ),
            UserWarning,
            stacklevel=2,
        )
    return ok, V_int, V_ref


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


def normalize_solver_mode(mode: str | None) -> str:
    """Normalize solver mode values to canonical form.
    
    Args:
        mode (str | None): Solver mode string.
        
    Returns:
        str: Canonical mode string ("overwrite" for override/default/None).
    """
    if mode in ("override", "default", None):
        return "overwrite"
    return str(mode)


def ensure_list(value: object | None, *, name: str, item_desc: str) -> list:
    """Ensure a value is a list (or empty), raising a descriptive error otherwise.
    
    Args:
        value (object | None): Value to check.
        name (str): Parameter name for error messages.
        item_desc (str): Description of items for error messages.
        
    Returns:
        list: The value as a list, or empty list if None.
        
    Raises:
        TypeError: If value is not a list or None.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of {item_desc}")
    return value
