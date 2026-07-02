"""Simple metrics for the heat-exhaust challenge."""

from fusdb import relation


@relation(
    name="Heat exhaust metric PB_over_R",
    tags=("power_exhaust", "tokamak"),
    outputs="PB_over_R",
)
def calc_PB_over_R(P_sep, B0, R):
    """Calculate P_sep*B0/R0, which scales roughly the same as the parallel heat flux density entering the scrape-off-layer.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_sep * B0 / R


@relation(
    name="Heat exhaust metric PBpRnSq",
    tags=("power_exhaust", "tokamak"),
    outputs="PBpRnSq",
)
def calc_PBpRnSq(P_sep, B0, qstar, R, n_e_avg):
    """Calculate P_sep * B_pol / (R * n^2), which scales roughly the same as the impurity fraction required for detachment.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (P_sep * (B0 / qstar) / R) / (n_e_avg**2.0)
