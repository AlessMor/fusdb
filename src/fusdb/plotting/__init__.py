"""Reusable plotters and table renderers for fusdb.

Each module owns one representation used across the example notebooks:

* :mod:`fusdb.plotting.reactivity`     -- fusion reactivity curves
* :mod:`fusdb.plotting.atomic_physics` -- atomic & molecular rate-coefficient curves (Bokeh)
* :mod:`fusdb.plotting.profiles`       -- radial plasma profiles
* :mod:`fusdb.plotting.relation_graph` -- relation/variable network graph
* :mod:`fusdb.plotting.curves`         -- generic x-y line/scan overlays
* :mod:`fusdb.plotting.comparison`     -- grouped-bar metric comparison
* :mod:`fusdb.plotting.maps`           -- 2-D parameter maps
* :mod:`fusdb.plotting.popcon`         -- POPCON contour maps of popcon-mode scans
* :mod:`fusdb.plotting.tables`         -- HTML/plain-text variable tables

Every plotter accepts an optional ``ax`` and returns the matplotlib ``Axes``,
so plots compose into figures the caller already owns.

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
    "plot_metric_comparison": "comparison",
    "plot_curves": "curves",
    "figure_to_html": "export",
    "plot_parameter_map": "maps",
    "plot_popcon": "popcon",
    "plot_profile_grid": "profiles",
    "plot_profiles": "profiles",
    "default_reactivities": "reactivity",
    "plot_reactivity": "reactivity",
    "bokeh_relation_graph": "relation_graph",
    "build_relation_graph": "relation_graph",
    "build_relation_node_graph": "relation_graph",
    "build_variable_relation_graph": "relation_graph",
    "plot_relation_graph": "relation_graph",
    "relation_graph_html": "relation_graph",
    "variables_table": "tables",
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
