"""Power exhaust relations defined once."""

from typing import Any

from fusdb import relation
@relation(
    name='P_sep ratio',
    tags=('power_exhaust',),
    
    outputs='P_sep_over_R',
)
def p_sep_ratio(P_sep: float, R: float) -> Any:
    """Return the P_sep / R ratio."""
    return P_sep / R


########################################
@relation(
    name='P_sep metric',
    tags=('power_exhaust',),
    
    outputs='P_sep_B_over_q95AR',
)
def p_sep_metric(P_sep: float, B0: float, q95: float, A: float, R: float) -> Any:
    """Return the P_sep * B0 / (q95 * A * R) metric."""
    return P_sep * B0 / (q95 * A * R)
