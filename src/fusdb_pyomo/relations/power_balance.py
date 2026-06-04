"""Power balance relations for heating, loss power, and stored energy."""

from __future__ import annotations

from typing import Any

from fusdb_pyomo import relation


# TODO(high): CHECK THE DEFINITIONS FOR P_LOSS AND ADD REFERENCES
@relation(
    name='Total plasma heating',
    tags=('power_balance',),
    outputs='P_heating',
)
def total_plasma_heating(P_ohmic: float, P_charged: float, P_aux: float) -> Any:
    """Return total plasma heating.

    Args:
        P_ohmic: Ohmic heating power.
        P_charged: Charged-particle fusion heating power.
        P_aux: Auxiliary heating power.

    Returns:
        Total plasma heating power.
    """
    # Sum every heating channel that deposits power in the plasma.
    return P_ohmic + P_charged + P_aux


########################################
@relation(
    name='Loss power to SOL and core radiation',
    tags=('power_balance',),
    outputs='P_loss_explicit',
)
def loss_power_to_exhaust(P_sep: float, P_rad: float) -> Any:
    """Return loss power to exhaust.

    Args:
        P_sep: Power crossing the separatrix.
        P_rad: Core radiated power.

    Returns:
        Total plasma loss power.
    """
    # Combine transport and radiative loss channels.
    return P_sep + P_rad


########################################
@relation(
    name='Power balance',
    tags=('power_balance',),
    outputs='P_loss',
)
def power_balance_simple(P_heating: float) -> Any:
    """Return loss power from simple power balance.

    Args:
        P_heating: Total plasma heating power.

    Returns:
        Total plasma loss power.
    """
    # Enforce steady-state equality between input and lost power.
    return P_heating


########################################
@relation(
    name='Thermal stored energy',
    tags=('power_balance', 'plasma'),
    outputs='W_th',
)
def thermal_stored_energy(p_th: float, V_p: float) -> Any:
    """Return thermal stored energy.

    Args:
        p_th: Thermal pressure.
        V_p: Plasma volume.

    Returns:
        Thermal stored energy.
    """
    # Convert pressure-volume content into thermal energy.
    return 1.5 * p_th * V_p


########################################
@relation(
    name='Energy confinement time',
    tags=('power_balance', 'plasma'),
    outputs='tau_E',
)
def energy_confinement_time(W_th: float, P_loss: float) -> Any:
    """Return energy confinement time.

    Args:
        W_th: Thermal stored energy.
        P_loss: Total plasma loss power.

    Returns:
        Energy confinement time.
    """
    # Divide stored energy by loss power to get the confinement timescale.
    return W_th / P_loss


# TODO(med): from PROCESS, add balance for ions and electrons (constraints.py)
