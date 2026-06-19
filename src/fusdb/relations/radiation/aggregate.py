"""Total radiated power relations."""
from typing import Any

from fusdb import relation

@relation(
    name='Total radiated power',
    tags=('power_balance',),
    outputs='P_rad',
)
def total_radiated_power(
    P_brem: float,
    P_line: float,
    P_sync: float,
 ) -> Any:
    """Return total radiated power from bremsstrahlung, line, and synchrotron radiation."""
    return P_brem + P_line + P_sync
