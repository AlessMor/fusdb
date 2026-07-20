"""Backend-neutral plotting data and explicit Matplotlib/Bokeh renderers.

Each module owns one representation used across the example notebooks:

* :mod:`fusdb.plotting.data`       -- ``CurveSet``, ``FieldMap``, and ``TableData``
* :mod:`fusdb.plotting.renderers`  -- explicit Matplotlib/Bokeh renderers
* :mod:`fusdb.plotting.reactivity` -- fusion reactivity curve-data discovery
* :mod:`fusdb.plotting.atomic_physics` -- interactive atomic-rate explorer
* :mod:`fusdb.plotting.profiles`   -- radial profile curve-data builder
* :mod:`fusdb.plotting.popcon`     -- POPCON field-map builder
* :mod:`fusdb.plotting.tables`     -- variable-table preparation/rendering

Submodules are imported lazily (PEP 562): the matplotlib/bokeh plotters need
the ``plotting`` extra, while :mod:`fusdb.plotting.tables` is dependency-free
and is imported by the core package -- accessing a plotter name here must not
drag matplotlib into every ``import fusdb``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public name -> owning submodule; resolved on first attribute access.
_EXPORTS = {
    "Curve": "data",
    "CurveSet": "data",
    "FieldMap": "data",
    "TableCell": "data",
    "TableData": "data",
    "figure_to_html": "export",
    "bokeh_curve_set": "renderers",
    "plot_curve_set": "renderers",
    "plot_field_map": "renderers",
    "popcon_field_map": "popcon",
    "profile_curves": "profiles",
    "reactivity_curves": "reactivity",
    "bokeh_relation_graph": "relation_graph",
    "build_relation_graph": "relation_graph",
    "build_relation_node_graph": "relation_graph",
    "build_variable_relation_graph": "relation_graph",
    "plot_relation_graph": "relation_graph",
    "relation_graph_html": "relation_graph",
    "variable_table_data": "tables",
    "render_table": "tables",
    "SolvedColumn": "tables",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
