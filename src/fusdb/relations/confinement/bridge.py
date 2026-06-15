"""Bridge relations for confinement, power balance, and density scalings.

Drop this file into a relations package that is imported during relation
registration, for example ``src/fusdb/relations/confinement/bridge.py`` or
another imported relations module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb import relation


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
    name="Line averaged density from average density",
    tags=("plasma", "confinement", "tokamak"),
    outputs="n_la",
)
def line_averaged_density_from_average_density(n_avg: float) -> float:
    """Approximate line-averaged density from volume-averaged density.

    NOTE: This is a temporary bridge so confinement scalings that require
    ``n_la`` can be reached when only ``n_avg`` is supplied.  It should be
    replaced by a proper line-average relation from a density profile and
    geometry, or by reactor-specific profile-shape information.

    Args:
        n_avg: Average plasma density.

    Returns:
        Approximate line-averaged density.
    """
    return n_avg
