"""Auxiliary power relations."""

from __future__ import annotations

from fusdb.relation_class import Relation_decorator as Relation
@Relation(
    name="Total auxiliary power",
    output="P_aux",
    tags=("power_exhaust", "auxiliary"),
)
def auxiliary_power(P_NBI: float, P_ICRF: float, P_LHCD: float) -> float:
    """Return total auxiliary power from injected sources."""
    return P_NBI + P_ICRF + P_LHCD
# TODO(med): specify launched and absorbed auxiliary power

# TODO(low): specify fraction of P_aux used for heating and for current drive
