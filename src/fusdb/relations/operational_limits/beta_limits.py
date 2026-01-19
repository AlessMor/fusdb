"""Operational beta limits."""

from __future__ import annotations

from fusdb.reactor_class import Reactor


@Reactor.relation(
    ("plasma", "tokamak"),
    name="Troyon beta limit",
    output="beta_limit",
    constraints=("a != 0", "B0 != 0", "I_p != 0"),
)
def troyon_beta_limit(a: float, B0: float, I_p: float) -> float:
    """Approximate Troyon limit: beta (fraction) = 0.028 * I_p / (a * B0)."""
    I_p_MA = I_p / 1e6
    return 0.028 * I_p_MA / (a * B0)
