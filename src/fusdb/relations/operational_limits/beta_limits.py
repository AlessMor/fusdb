"""Operational beta limit relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name='Troyon beta limit',
    tags=('plasma', 'tokamak'),
    
    outputs='beta_limit',
)
def troyon_beta_limit(a: float, B0: float, I_p: float) -> Any:
    """Approximate Troyon limit: beta (fraction) = 0.028 * I_p / (a * B0)."""
    I_p_MA = I_p / 1e6
    return 0.028 * I_p_MA / (a * B0)


@relation(
    name='Troyon margin',
    tags=('plasma', 'tokamak', 'constraint'),
    outputs='troyon_margin',
)
def troyon_margin(beta_T: float, beta_limit: float) -> Any:
    """Return Troyon margin (<=0 satisfied)."""
    return beta_T - beta_limit


# --- Normalised-beta upper-limit (beta_N_max) scalings (PROCESS physics.py) ----
# The i_beta_norm_max enum split into one relation per scaling, all producing
# beta_norm_max and gated in variables.yaml (Wesson default).
_BN_TAGS = ("plasma", "tokamak", "process")


@relation(name="Normalised beta limit (Wesson)", tags=_BN_TAGS, outputs="beta_norm_max")
def calculate_beta_norm_max_wesson(internal_inductivity: float) -> Any:
    """Wesson normalised-beta upper limit as a fraction: 0.04 * li.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.04 * internal_inductivity


@relation(name="Normalised beta limit (original)", tags=_BN_TAGS, outputs="beta_norm_max")
def calculate_beta_norm_max_original(eps: float) -> Any:
    """Original normalised-beta upper limit scaling, returned as a fraction.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.01 * 2.7 * (1.0 + 5.0 * eps**3.5)


@relation(name="Normalised beta limit (Menard)", tags=_BN_TAGS, outputs="beta_norm_max")
def calculate_beta_norm_max_menard(eps: float) -> Any:
    """Menard normalised-beta upper limit as a fraction (spherical tokamak, f_BS ~ 50%).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.01 * (3.12 + 3.5 * eps**1.7)


@relation(name="Normalised beta limit (Tholerus)", tags=_BN_TAGS, outputs="beta_norm_max")
def calculate_beta_norm_max_thloreus(
    c_beta: float, pres_plasma_on_axis: float, p_th: float
) -> Any:
    """Tholerus normalised-beta upper limit as a fraction (STEP flat-top operational space);
    ``p_th`` is the volume-averaged plasma pressure.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    fp = pres_plasma_on_axis / p_th
    return 0.01 * (3.7 + ((c_beta / fp) * (12.5 - 3.5 * fp)))


@relation(name="Normalised beta limit (Stambaugh)", tags=_BN_TAGS, outputs="beta_norm_max")
def calculate_beta_norm_max_stambaugh(f_BS: float, kappa: float, A: float) -> Any:
    """Stambaugh normalised-beta upper limit as a fraction (steady-state tokamak equilibria).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return 0.01 * (
        f_BS
        * 10
        * (-0.7748 + (1.2869 * kappa) - (0.2921 * kappa**2) + (0.0197 * kappa**3))
        / (A**0.5523 * np.tanh((1.8524 + (0.2319 * kappa)) / A**0.6163))
    )
