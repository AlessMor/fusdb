"""Operational beta limits."""

from __future__ import annotations

from fusdb.relation_util import relation
@relation(
    name="Troyon beta limit",
    output="beta_limit",
    tags=("plasma", "tokamak"),
    constraints=("a != 0", "B0 != 0", "I_p != 0"),
)
def troyon_beta_limit(a: float, B0: float, I_p: float) -> float:
    """Approximate Troyon limit: beta (fraction) = 0.028 * I_p / (a * B0)."""
    I_p_MA = I_p / 1e6
    return 0.028 * I_p_MA / (a * B0)


########################################
@relation(
    name="Troyon margin",
    output="troyon_margin",
    tags=("plasma", "tokamak", "constraint"),
    soft_constraints=("troyon_margin <= 0",),
)
def troyon_margin(beta_T: float, beta_limit: float) -> float:
    """Return Troyon margin (<=0 satisfied)."""
    return beta_T - beta_limit
