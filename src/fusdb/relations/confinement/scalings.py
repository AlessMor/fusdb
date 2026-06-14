"""Confinement scaling relations."""

from __future__ import annotations

from typing import Any

from fusdb import relation


@relation(
    name="tau_E_iter_ipb98y2",
    tags=("confinement", "tokamak", "h_mode"),
    outputs="tau_E",
)
def tau_E_iter_ipb98y2(
    H98_y2: float,
    I_p: float,
    B0: float,
    n_la: float,
    P_loss: float,
    R: float,
    kappa: float,
    A: float,
    afuel: float,
) -> Any:
    """Return the ITER IPB98(y,2)-style confinement time scaling.

    Uses MW, MA, and 1e19 m^-3 normalized inputs internally, while FusDB stores
    canonical SI values for power/current/density.
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return H98_y2 * (
        0.0365
        * I_p_MA**0.97
        * B0**0.08
        * dnla19**0.41
        * P_loss_MW ** (-0.63)
        * R**1.93
        * kappa**0.67
        * A ** (-0.23)
        * afuel**0.2
    )


def iter_ipb98y_confinement_time(
    I_p: float,
    B0: float,
    n_la: float,
    P_loss: float,
    R: float,
    kappa: float,
    A: float,
    afuel: float,
) -> float:
    """Undecorated helper matching the relation without H98_y2."""
    return tau_E_iter_ipb98y2.func(1.0, I_p, B0, n_la, P_loss, R, kappa, A, afuel)


tau_E_ipb98y = iter_ipb98y_confinement_time
