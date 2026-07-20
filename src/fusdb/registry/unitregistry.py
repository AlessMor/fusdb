"""Unit conversion helpers backed by pint.

Values are converted once at the ingestion boundary (``Variable``); nothing in
the solver hot path touches units, so the pint registry is built lazily on the
first conversion.
"""

from __future__ import annotations

from typing import Any
import numpy as np

_UNIT_REGISTRY = None


def unit_registry():
    global _UNIT_REGISTRY
    if _UNIT_REGISTRY is None:
        import pint
        _UNIT_REGISTRY = pint.UnitRegistry()
    return _UNIT_REGISTRY


def _normalize_unit(text: str | None) -> str:
    unit = "" if text is None else str(text).strip()
    if unit in {"", "1", "dimensionless", "none", "None"}:
        return "dimensionless"
    return unit.replace("^", "**")


def convert_value(value: Any, *, from_unit: str | None, to_unit: str | None) -> Any:
    """Convert one scalar or array-like value between units."""
    if value is None:
        return None
    dst = _normalize_unit(to_unit)
    if hasattr(value, "to") and hasattr(value, "magnitude"):
        converted = value.to(dst).magnitude
    else:
        src = _normalize_unit(from_unit)
        if src == dst:
            return value
        try:
            converted = (np.asarray(value) * unit_registry()(src)).to(dst).magnitude
        except Exception as exc:
            raise ValueError(f"Cannot convert unit {src!r} -> {dst!r}: {exc}") from exc
    arr = np.asarray(converted)
    return float(arr) if arr.ndim == 0 else arr
