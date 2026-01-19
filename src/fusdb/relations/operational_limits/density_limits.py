"""Operational density limits."""

from __future__ import annotations

import sympy as sp

from fusdb.reactor_class import Reactor
from fusdb.relation_util import nonzero, symbol


@Reactor.relation(
    ("plasma", "tokamak"),
    name="Greenwald density limit",
    output="n_GW",
    constraints=("a != 0"),
)
def greenwald_density_limit(I_p: float, a: float) -> float:
    """Return Greenwald density limit in 1/m^3 for tokamaks."""
    I_p_MA = I_p / 1e6
    return 1e20 * I_p_MA / (sp.pi * a**2)

@Reactor.relation(
    ("plasma", "tokamak"),
    name="Greenwald density fraction",
    output="f_GW", # type: ignore[arg-type]
    constraints=("n_GW != 0",)
)
def greenwald_density_fraction(n_GW: float, n_avg: float) -> float:
    """Return fraction of Greenwald density limit."""
    f_GW =  n_avg / n_GW
    return f_GW

@Reactor.relation(
    ("plasma", "stellarator"),
    name="Sudo density limit",
    output="n_SUDO",
    constraints=("R != 0", "a != 0"),
)
def sudo_density_limit(P_loss: float, B0: float, R: float, a: float) -> float:
    """Return Sudo density limit in 1/m^3 for stellarators."""
    P_loss_MW = P_loss / 1e6
    return 1e20 * 0.25 * P_loss_MW * B0 / (R * a**2)

# TODO(low): from PROCESS - physics/calculate_density_limit
