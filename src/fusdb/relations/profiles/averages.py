"""Profile average relations."""

from fusdb import relation


@relation(
    name="Line averaged density from average density",
    tags=("plasma", "confinement", "tokamak"),
    outputs="n_la",
)
def line_averaged_density_from_average_density(n_avg: float) -> float:
    """Approximate line-averaged density from volume-averaged density.

    NOTE: This is a temporary bridge so confinement scalings that require
    ``n_la`` can be reached when only ``n_avg`` is supplied.  It should be
    replaced by a proper line-average relation from a density profile and
    geometry, or by reactor-specific profile-shape information.

    Args:
        n_avg: Average plasma density.

    Returns:
        Approximate line-averaged density.
    """
    return n_avg
