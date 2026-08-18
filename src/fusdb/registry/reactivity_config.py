"""Fixed numerical settings for tabulated reactivity evaluation.

These are package implementation constants rather than runtime configuration.
The ``SimpleNamespace`` preserves the existing attribute-style internal API
without introducing a FusDB configuration class.
"""

from types import SimpleNamespace

REACTIVITY_TABLES = SimpleNamespace(
    energy_grid_start_log10_kev=0.0,
    energy_grid_stop_log10_kev=5.0,
    energy_grid_num_points=1000,
    allowed_interpolation_kinds=frozenset(
        {"pchip", "linear", "nearest", "zero", "slinear", "quadratic", "cubic"}
    ),
)

__all__ = ["REACTIVITY_TABLES"]
