"""POPCON plots: contour maps of a 2-D popcon-mode scan result.

Draws the classic plasma-operating-contour figure from the ``"popcon"``
payload produced by :mod:`fusdb.modes.popcon`: an optional filled field, one
labelled iso-contour set per requested output, and a grey mask over grid
points that did not solve.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ..registry import VARIABLES
from .data import FieldMap


def _field_label(name: str) -> str:
    """``name [unit]`` from the registry; plain ``name`` for ad-hoc fields."""
    if name in VARIABLES:
        unit = VARIABLES.get(name).unit
        if unit and unit != "dimensionless":
            return f"{name} [{unit}]"
    return name


def popcon_field_map(result: Mapping[str, Any], *, title: str | None = None) -> FieldMap:
    """Convert a POPCON result payload into backend-neutral map data."""
    payload = result.get("popcon", result)
    x = np.asarray(payload["x"]["values"], dtype=float)
    y = np.asarray(payload["y"]["values"], dtype=float)
    if x.ndim == y.ndim == 1:
        x, y = np.meshgrid(x, y)
    return FieldMap(
        x,
        y,
        payload["fields"],
        xlabel=_field_label(payload["x"]["name"]),
        ylabel=_field_label(payload["y"]["name"]),
        title=title,
        metadata={
            "mask": ~np.asarray(payload["success"], dtype=bool),
            "field_labels": {name: _field_label(name) for name in payload["fields"]},
        },
    )
