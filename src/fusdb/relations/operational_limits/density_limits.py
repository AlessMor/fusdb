"""Operational density limits."""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb import relation



@relation(
    name='Greenwald density limit',
    tags=('plasma', 'tokamak'),
    
    outputs='n_GW',
)
def greenwald_density_limit(I_p: float, a: float) -> Any:
    """Return Greenwald density limit in 1/m^3 for tokamaks."""
    I_p_MA = I_p / 1e6
    return 1e20 * I_p_MA / (np.pi * a**2)

########################################
@relation(
    name='Greenwald density fraction',
    tags=('plasma', 'tokamak'),
    
    outputs='f_GW',
)
def greenwald_density_fraction(n_GW: float, n_avg: float) -> Any:
    """Return fraction of Greenwald density limit."""
    f_GW =  n_avg / n_GW
    return f_GW

########################################
@relation(
    name='Greenwald margin',
    tags=('plasma', 'tokamak', 'constraint'),
    outputs='greenwald_margin',
)
def greenwald_margin(n_avg: float, n_GW: float) -> Any:
    """Return Greenwald margin (<=0 satisfied)."""
    return n_avg - n_GW

########################################
@relation(
    name='Sudo density limit',
    tags=('plasma', 'stellarator'),
    
    outputs='n_SUDO',
)
def sudo_density_limit(P_loss: float, B0: float, R: float, a: float) -> Any:
    """Return Sudo density limit in 1/m^3 for stellarators."""
    P_loss_MW = P_loss / 1e6
    return 1e20 * 0.25 * P_loss_MW * B0 / (R * a**2)


########################################
@relation(
    name='Sudo margin',
    tags=('plasma', 'stellarator', 'constraint'),
    outputs='sudo_margin',
)
def sudo_margin(n_avg: float, n_SUDO: float) -> Any:
    """Return Sudo margin (<=0 satisfied)."""
    return n_avg - n_SUDO


# TODO(low): from PROCESS - physics/calculate_density_limit
