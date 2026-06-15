"""L-H transition relations."""

from __future__ import annotations

from typing import Any

from fusdb import relation

# TODO: add more L-H transition relations

@relation(
    name='L-H transition threshold power',
    tags=('confinement', 'h_mode', 'constraint'),
    outputs='P_LH',
)
def lh_transition_power(n_avg: float, B0: float, A_p: float) -> Any:
    """Return the L-H transition threshold power using a Martin-2008 style scaling.

    Args:
        n_avg: Line-averaged density [1/m^3].
        B0: Toroidal magnetic field [T].
        A_p: Plasma surface area [m^2].

    Returns:
        L-H transition threshold power [W].
    """
    n20 = n_avg / 1e20
    # P_LH [MW] = 0.0488 * n20^0.717 * B0^0.803 * A_p^0.941
    return 1e6 * 0.0488 * (n20 ** 0.717) * (B0 ** 0.803) * (A_p ** 0.941)
