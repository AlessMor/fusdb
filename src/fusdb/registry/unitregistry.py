"""Unit conversion helpers with a small no-pint fallback."""

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


_FALLBACK_FACTORS = {
    ("MW", "W"): 1e6,
    ("GW", "W"): 1e9,
    ("kW", "W"): 1e3,
    ("GJ", "J"): 1e9,
    ("MJ", "J"): 1e6,
    ("kJ", "J"): 1e3,
    ("MA", "A"): 1e6,
    ("kA", "A"): 1e3,
    ("keV", "keV"): 1.0,
    ("amu", "amu"): 1.0,
    ("s", "s"): 1.0,
    ("m", "m"): 1.0,
    ("T", "T"): 1.0,
    ("MW/m", "W/m"): 1e6,
    ("MW*T/m", "W*T/m"): 1e6,
    ("1e20/m**3", "1/m**3"): 1e20,
    ("1e20 m**-3", "1/m**3"): 1e20,
    ("1e20/m^3", "1/m**3"): 1e20,
}


def convert_value(value: Any, *, from_unit: str | None, to_unit: str | None) -> Any:
    """Convert one scalar or array-like value between units."""
    if value is None:
        return None
    dst = "" if to_unit is None else str(to_unit).strip()
    if dst in {"", "1", "dimensionless", "none", "None"}:
        dst = "dimensionless"
    else:
        dst = dst.replace("^", "**")
    if hasattr(value, "to") and hasattr(value, "magnitude"):
        converted = value.to(dst).magnitude
    else:
        src = "" if from_unit is None else str(from_unit).strip()
        if src in {"", "1", "dimensionless", "none", "None"}:
            src = "dimensionless"
        else:
            src = src.replace("^", "**")
        if src == dst:
            return value
        try:
            converted = (np.asarray(value) * unit_registry()(src)).to(dst).magnitude
        except Exception:
            key = (src, dst)
            if key not in _FALLBACK_FACTORS:
                raise RuntimeError(f"Unit conversion requires pint for {src!r} -> {dst!r}.")
            converted = np.asarray(value) * _FALLBACK_FACTORS[key]
    arr = np.asarray(converted)
    return float(arr) if arr.ndim == 0 else arr
