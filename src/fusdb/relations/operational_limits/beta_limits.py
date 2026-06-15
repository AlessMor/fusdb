"""Operational beta limits."""

from __future__ import annotations

from typing import Any

from fusdb import relation
@relation(
    name='Troyon beta limit',
    tags=('plasma', 'tokamak'),
    
    outputs='beta_limit',
)
def troyon_beta_limit(a: float, B0: float, I_p: float) -> Any:
    """Approximate Troyon limit: beta (fraction) = 0.028 * I_p / (a * B0)."""
    I_p_MA = I_p / 1e6
    return 0.028 * I_p_MA / (a * B0)


########################################
@relation(
    name='Troyon margin',
    tags=('plasma', 'tokamak', 'constraint'),
    outputs='troyon_margin',
)
def troyon_margin(beta_T: float, beta_limit: float) -> Any:
    """Return Troyon margin (<=0 satisfied)."""
    return beta_T - beta_limit
