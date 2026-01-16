"""Auxiliary power relations."""

from __future__ import annotations

from fusdb.reactor_class import Reactor


@Reactor.relation(
    ("power_exhaust", "auxiliary"),
    name="Total auxiliary power",
    output="P_aux",
    solve_for=("P_aux",),
)
def auxiliary_power(P_NBI: float, P_ICRF: float, P_LHCD: float) -> float:
    """Return total auxiliary power from injected sources."""
    return P_NBI + P_ICRF + P_LHCD


# TODO(med): specify launched and absorbed auxiliary power