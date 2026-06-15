"""Auxiliary power relations."""

from __future__ import annotations

from typing import Any

from fusdb import relation
@relation(
    name='Total auxiliary power',
    tags=('power_exhaust', 'auxiliary'),
    outputs='P_aux',
)
def auxiliary_power(P_NBI: float, P_ICRF: float, P_LHCD: float) -> Any:
    """Return total auxiliary power from injected sources."""
    return P_NBI + P_ICRF + P_LHCD
# TODO(med): specify launched and absorbed auxiliary power

# TODO(low): specify fraction of P_aux used for heating and for current drive
