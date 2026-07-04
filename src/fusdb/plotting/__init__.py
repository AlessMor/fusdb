"""Reusable matplotlib plotters for fusdb (install the ``plotting`` extra).

Each module owns one representation used across the example notebooks:

* :mod:`fusdb.plotting.reactivity`     -- fusion reactivity curves
* :mod:`fusdb.plotting.atomic_physics` -- atomic & molecular rate-coefficient curves (Bokeh)
* :mod:`fusdb.plotting.profiles`       -- radial plasma profiles
* :mod:`fusdb.plotting.relation_graph` -- relation/variable network graph
* :mod:`fusdb.plotting.curves`         -- generic x-y line/scan overlays
* :mod:`fusdb.plotting.comparison`     -- grouped-bar metric comparison
* :mod:`fusdb.plotting.maps`           -- 2-D parameter maps
* :mod:`fusdb.plotting.popcon`         -- POPCON contour maps of popcon-mode scans

Every plotter accepts an optional ``ax`` and returns the matplotlib ``Axes``,
so plots compose into figures the caller already owns.
"""

from .comparison import plot_metric_comparison
from .curves import plot_curves
from .export import figure_to_html
from .maps import plot_parameter_map
from .popcon import plot_popcon
from .profiles import plot_profile_grid, plot_profiles
from .reactivity import default_reactivities, plot_reactivity
from .relation_graph import (
    bokeh_relation_graph,
    build_relation_graph,
    build_relation_node_graph,
    build_variable_relation_graph,
    plot_relation_graph,
    relation_graph_html,
)

__all__ = [
    "plot_reactivity",
    "default_reactivities",
    "plot_profiles",
    "plot_profile_grid",
    "build_relation_graph",
    "build_relation_node_graph",
    "build_variable_relation_graph",
    "plot_relation_graph",
    "bokeh_relation_graph",
    "relation_graph_html",
    "plot_curves",
    "plot_metric_comparison",
    "plot_parameter_map",
    "plot_popcon",
    "figure_to_html",
]
