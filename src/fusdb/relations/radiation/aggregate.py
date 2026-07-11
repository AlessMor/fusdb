"""Total radiated power relations."""
from typing import Any

import numpy as np

from fusdb.relation import relation

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


@relation(
    name='Core radiated power fraction',
    tags=('power_balance',),
    outputs='f_rad',
)
def calc_f_rad_core(P_rad: float, P_in: float) -> Any:
    """Return the core radiated power fraction P_rad / P_in.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_rad / P_in


@relation(
    name='Minimum radiated power from fraction',
    tags=('power_balance',),
    outputs='min_P_radiation',
)
def calc_min_P_radiation_from_fraction(minimum_core_radiated_fraction: float, P_in: float) -> Any:
    """Set the minimum radiated power as a fraction of the total input power.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return minimum_core_radiated_fraction * P_in


@relation(
    name='Minimum radiated power from LH factor',
    tags=('power_balance', 'tokamak'),
    outputs='min_P_radiation',
)
def calc_min_P_radiation_from_LH_factor(maximum_P_LH_factor_for_P_SOL: float, P_LH: float, P_in: float) -> Any:
    """Set the minimum radiated power so P_sol stays below a multiple of the L-H threshold.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return np.maximum(P_in - maximum_P_LH_factor_for_P_SOL * P_LH, 0.0)
