"""Scrape-off-layer separatrix density relations."""

from fusdb.relation import relation


@relation(
    name="Separatrix electron density from average",
    tags=("power_exhaust", "tokamak"),
    outputs="n_sep",
)
def calc_separatrix_electron_density(nesep_over_nebar, n_e_avg):
    """Calculate the separatrix electron density, assuming a constant ratio to the average electron density.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return nesep_over_nebar * n_e_avg
