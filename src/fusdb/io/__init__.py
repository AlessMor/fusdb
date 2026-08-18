"""FusDB input/output: result persistence and display tables."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .results import load_result, save_result

_TABLE_EXPORTS = {"SolvedColumn", "TableCell", "TableData", "render_table", "variable_table_data"}

__all__ = ["load_result", "save_result", *_TABLE_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _TABLE_EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(".tables", __name__), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_TABLE_EXPORTS})
