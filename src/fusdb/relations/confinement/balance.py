"""Energy confinement balance relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name="Plasma stored energy from averages",
    tags=("plasma", "confinement"),
    outputs="W_th",
)
def calc_plasma_stored_energy(p_th: float, V_p: float) -> Any:
    """Return thermal stored energy from volume-averaged thermal pressure.

    The relation keeps the historical name so existing reactor files that
    include ``Plasma stored energy from averages`` keep working, but the stored
    energy definition is the profile-consistent volume integral:
    ``W_th = 3/2 * <p_th>_V * V_p``.
    """
    return 1.5 * p_th * V_p


@relation(
    name="Energy confinement balance",
    tags=("confinement", "power_balance", "plasma"),
)
def energy_confinement_balance(W_th: float, P_loss: float, tau_E: float) -> Any:
    """Return the normalized residual for ``W_th = P_loss * tau_E``.

    This is intentionally an outputless residual relation rather than
    ``tau_E = W_th / P_loss``.  That lets it be active at the same time as a
    confinement-scaling relation that also determines ``tau_E``.  In reconcile
    mode the two relations then jointly enforce consistency instead of one
    relation replacing or shadowing the other.

    Args:
        W_th: Thermal stored energy.
        P_loss: Plasma loss power.
        tau_E: Energy confinement time.

    Returns:
        Dimensionless residual, equal to zero when the balance is satisfied.
    """
    # Reject non-physical power and confinement values before forming the balance.
    for name, value in (("P_loss", P_loss), ("tau_E", tau_E)):
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    lhs = np.asarray(W_th, dtype=float)
    rhs = np.asarray(P_loss, dtype=float) * np.asarray(tau_E, dtype=float)
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


@relation(
    name='Thermal stored energy',
    tags=('plasma',),
    outputs='W_th',
)
def thermal_stored_energy(p_th: float, V_p: float) -> float:
    """Return thermal stored energy from pressure and plasma volume.

    Args:
        p_th: Volume-averaged thermal pressure.
        V_p: Plasma volume.

    Returns:
        Thermal stored energy.
    """
    return 1.5 * p_th * V_p
