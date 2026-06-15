"""Execution modes for RelationSystem."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

_MODE_MODULES = {
    "verify": "fusdb.modes.verify",
    "ordered": "fusdb.modes.ordered",
    "reconcile": "fusdb.modes.reconcile",
    "optimize": "fusdb.modes.optimize",
}


def get_mode(name: str) -> Callable[..., dict[str, Any]]:
    try:
        module_name = _MODE_MODULES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown mode {name!r}.") from exc
    return import_module(module_name).run
