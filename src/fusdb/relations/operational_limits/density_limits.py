"""Operational density limits."""

from __future__ import annotations

import sympy as sp

from fusdb.relation_util import relation
@relation(
    name="Greenwald density limit",
    output="n_GW",
    tags=("plasma", "tokamak"),
    constraints=("a != 0",),
)
def greenwald_density_limit(I_p: float, a: float) -> float:
    """Return Greenwald density limit in 1/m^3 for tokamaks."""
    I_p_MA = I_p / 1e6
    return 1e20 * I_p_MA / (sp.pi * a**2)

########################################
@relation(
    name="Greenwald density fraction",
    output="f_GW",
    tags=("plasma", "tokamak"),
    constraints=("n_GW != 0",),
)
def greenwald_density_fraction(n_GW: float, n_avg: float) -> float:
    """Return fraction of Greenwald density limit."""
    f_GW =  n_avg / n_GW
    return f_GW

########################################
@relation(
    name="Greenwald margin",
    output="greenwald_margin",
    tags=("plasma", "tokamak", "constraint"),
    soft_constraints=("greenwald_margin <= 0",),
)
def greenwald_margin(n_avg: float, n_GW: float) -> float:
    """Return Greenwald margin (<=0 satisfied)."""
    return n_avg - n_GW

########################################
@relation(
    name="Sudo density limit",
    output="n_SUDO",
    tags=("plasma", "stellarator"),
    constraints=("R != 0", "a != 0"),
)
def sudo_density_limit(P_loss: float, B0: float, R: float, a: float) -> float:
    """Return Sudo density limit in 1/m^3 for stellarators."""
    P_loss_MW = P_loss / 1e6
    return 1e20 * 0.25 * P_loss_MW * B0 / (R * a**2)


########################################
@relation(
    name="Sudo margin",
    output="sudo_margin",
    tags=("plasma", "stellarator", "constraint"),
    soft_constraints=("sudo_margin <= 0",),
)
def sudo_margin(n_avg: float, n_SUDO: float) -> float:
    """Return Sudo margin (<=0 satisfied)."""
    return n_avg - n_SUDO


# TODO(low): from PROCESS - physics/calculate_density_limit
