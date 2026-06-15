"""Reusable matplotlib plotters for fusdb (install the ``plotting`` extra).

Each module owns one representation used across the example notebooks:

* :mod:`fusdb.plotting.reactivity`     -- fusion reactivity curves
* :mod:`fusdb.plotting.profiles`       -- radial plasma profiles
* :mod:`fusdb.plotting.relation_graph` -- relation/variable network graph
* :mod:`fusdb.plotting.curves`         -- generic x-y line/scan overlays
* :mod:`fusdb.plotting.comparison`     -- grouped-bar metric comparison
* :mod:`fusdb.plotting.maps`           -- 2-D parameter maps

Every plotter accepts an optional ``ax`` and returns the matplotlib ``Axes``,
so plots compose into figures the caller already owns.
"""

from .comparison import plot_metric_comparison
from .curves import plot_curves
from .export import figure_to_html
from .maps import plot_parameter_map
from .profiles import plot_profile_grid, plot_profiles
from .reactivity import default_reactivities, plot_reactivity
from .relation_graph import build_relation_graph, plot_relation_graph

__all__ = [
    "plot_reactivity",
    "default_reactivities",
    "plot_profiles",
    "plot_profile_grid",
    "build_relation_graph",
    "plot_relation_graph",
    "plot_curves",
    "plot_metric_comparison",
    "plot_parameter_map",
    "figure_to_html",
]
