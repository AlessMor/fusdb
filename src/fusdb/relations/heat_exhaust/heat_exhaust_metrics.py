"""Simple metrics for the heat-exhaust challenge."""

from typing import Any

import numpy as np

from fusdb.relation import relation


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


@relation(name="Mirror throat power flux", tags=("mirror", "power_exhaust"), outputs="q_throat")
def mirror_throat_power_flux(P_loss: Any, A_th: Any) -> Any:
    """Symmetric two-ended reduced end-load diagnostic.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return np.asarray(P_loss) / (2.0 * np.asarray(A_th))


@relation(name="Mirror collector power flux", tags=("mirror", "power_exhaust"), outputs="q_collector")
def mirror_collector_power_flux(q_throat: Any, collector_area_ratio: Any) -> Any:
    """Mirror end-loss flux diluted over an expanded collector area.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return np.asarray(q_throat) / np.asarray(collector_area_ratio)
