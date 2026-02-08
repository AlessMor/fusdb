"""Shared numeric and utility helpers for the fusdb package.

This module provides common utility functions for:
- Numeric comparisons with tolerances
- YAML file loading
- Tag normalization and comparison
- Country name normalization
"""
from __future__ import annotations

import math
from pathlib import Path
import yaml
import pycountry


def within_tolerance(
    a: object,
    b: object,
    *,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> bool:
    """Check if two values are equal within specified tolerances.
    
    Compares two values using both absolute and relative tolerance.
    The comparison is: |a - b| <= max(abs_tol, rel_tol * scale)
    where scale is max(|a|, |b|, 1.0).
    
    Args:
        a: First value (must be convertible to float).
        b: Second value (must be convertible to float).
        rel_tol: Relative tolerance (fraction of larger value).
        abs_tol: Absolute tolerance (in same units as values).
    
    Returns:
        True if values are within tolerance, False otherwise.
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
    """Convert value to float, returning None if conversion fails or non-finite."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_yaml(path: Path | str) -> dict:
    """Load and parse a YAML file into a dictionary.
    
    Args:
        path: Path to YAML file.

    Returns:
        Dictionary containing parsed YAML content.
        Returns empty dict if file is empty or contains only None.
        
    Raises:
        FileNotFoundError: If path doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_tag(tag: str | None) -> str:
    """Normalize a single tag for consistent comparison.
    
    Converts tag to lowercase and removes non-alphanumeric characters.
    
    Args:
        tag: Single tag string or None
        
    Returns:
        Normalized tag string (empty string if None/empty)
        
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
        tags: Tags as string, iterable of strings, or None
        
    Returns:
        Tuple of normalized tag strings
        
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
        country: Country name, code (ISO 3166), or alias.
    
    Returns:
        Official country name from pycountry, or the original string
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
    """Normalize solver mode values to canonical form."""
    if mode in ("override", "default", None):
        return "overwrite"
    return str(mode)


def ensure_list(value: object | None, *, name: str, item_desc: str) -> list:
    """Ensure a value is a list (or empty), raising a descriptive error otherwise."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of {item_desc}")
    return value
