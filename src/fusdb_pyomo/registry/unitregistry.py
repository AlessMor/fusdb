"""Pint-backed unit conversion helpers."""

from __future__ import annotations

from typing import Any
import numpy as np

_UNIT_REGISTRY = None


def unit_registry():
    """Return the shared Pint unit registry."""
    global _UNIT_REGISTRY
    if _UNIT_REGISTRY is None:
        try:
            import pint
        except ImportError as exc:
            raise RuntimeError("Unit conversion requires pint.") from exc
        _UNIT_REGISTRY = pint.UnitRegistry()
    return _UNIT_REGISTRY


def _unit_text(unit: str | None) -> str:
    text = "" if unit is None else str(unit).strip()
    if text in {"", "1", "dimensionless", "none", "None"}:
        return "dimensionless"
    return text.replace("^", "**")


def convert_value(value: Any, *, from_unit: str | None, to_unit: str | None) -> Any:
    """Convert a scalar or array value into the registry unit.

    Args:
        value: Plain number, array, or Pint quantity.
        from_unit: Input unit, ignored when ``value`` is already a Pint quantity.
        to_unit: Target canonical unit.

    Returns:
        Plain float or NumPy array in the target unit.
    """
    if value is None:
        return None
    dst = _unit_text(to_unit)

    if hasattr(value, "to") and hasattr(value, "magnitude"):
        converted = value.to(dst).magnitude
    else:
        src = _unit_text(from_unit)
        if src == dst:
            return value
        converted = (np.asarray(value) * unit_registry()(src)).to(dst).magnitude

    arr = np.asarray(converted)
    return float(arr) if arr.ndim == 0 else arr
