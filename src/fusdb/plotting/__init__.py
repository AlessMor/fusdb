"""Backend-neutral plotting and display data with lazy renderers.

Table preparation is dependency-light. Matplotlib and Bokeh backends are
loaded only when their exported names are accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Curve": "data",
    "CurveSet": "data",
    "FieldMap": "data",
    "TableCell": "tables",
    "TableData": "tables",
    "SolvedColumn": "tables",
    "render_table": "tables",
    "variable_table_data": "tables",
    "bokeh_curve_set": "bokeh",
    "plot_curve_set": "matplotlib",
    "plot_field_map": "matplotlib",
    "popcon_field_map": "popcon",
    "profile_curves": "profiles",
    "reactivity_curves": "reactivity",
    "bokeh_relation_graph": "relation_graph",
    "build_relation_graph": "relation_graph",
    "plot_relation_graph": "relation_graph",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
