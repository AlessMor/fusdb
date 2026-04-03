"""Radiated power relations."""

from __future__ import annotations

from fusdb.relation_util import relation
# NOTE: PROCESS and cfspopcon use radas radiation loss function Lz, that includes all sources of radiation in a formula
# P_i = n_i * n_e * Lz(Z_i, T_i).


@relation(
    name="Total radiated power",
    output="P_rad",
    tags=("power_balance",),
    constraints=("P_rad >= 0",),
)
def total_radiated_power(
    P_brem: float,
    P_line: float,
    P_sync: float,
) -> float:
    """Return total radiated power from bremsstrahlung, line, and synchrotron radiation."""
    return P_brem + P_line + P_sync
