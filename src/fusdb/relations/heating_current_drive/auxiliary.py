"""Auxiliary heating power relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name='Total auxiliary power',
    tags=('auxiliary_power',),
    outputs='P_aux',
)
def auxiliary_power_from_sources(P_NBI: float = 0.0, P_ICRF: float = 0.0, P_LHCD: float = 0.0, P_ECRH: float = 0.0) -> Any:
    """Return total auxiliary power from the heating channels a scenario declares.

    Each channel carries a signature default, making it an OPTIONAL CONTRIBUTOR
    rather than a constant: a declared channel is read from the namespace, an
    undeclared one is zero.  Which systems a machine HAS is a fact about that
    machine, so it belongs in its reactor file, not in a registry default that
    would claim the same for every device.

    Being all-optional, this relation stays out of the graph until at least one
    channel is supplied or derivable -- the guard is in the forward closure,
    because with no required inputs ``all(inp in known for inp in ())`` is
    vacuously true and it would otherwise fire from nothing.  See TODO.
    """
    return P_NBI + P_ICRF + P_LHCD + P_ECRH


@relation(
    name='Plasma loss power',
    tags=('auxiliary_power', 'power_balance', 'plasma'),
    outputs='P_loss',
)
def plasma_loss_power(P_charged: float, P_aux: float) -> Any:
    """Return steady-state plasma loss power from charged fusion and auxiliary heating."""
    return P_charged + P_aux


@relation(
    name='Steady-state input-loss power balance',
    tags=('auxiliary_power', 'power_balance', 'plasma'),
)
def steady_state_input_loss_balance(P_in: float, P_loss: float) -> Any:
    """Return the normalized residual for ``P_in = P_loss``.

    cfspopcon defines P_in as the confinement-scaling heating power
    (P_ohmic + P_alpha + P_aux) and identifies it with the loss power
    P_loss = P_SOL + P_rad,core under the steady-state assumption
    dW_th/dt = 0. This is an outputless residual, like
    ``energy_confinement_balance``, so it lets P_in and P_loss be inferred
    from each other without one relation shadowing the other's producer.
    """
    lhs = np.asarray(P_in, dtype=float)
    rhs = np.asarray(P_loss, dtype=float)
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


@relation(
    name='Absorbed auxiliary power',
    tags=('auxiliary_power',),
    outputs='P_aux_absorbed',
)
def auxiliary_power_absorbed_from_launched(P_aux_launched: float, fraction_of_external_power_coupled: float) -> Any:
    """Return absorbed auxiliary power from launched auxiliary power and coupling fraction.
    Adapted from cfspopcon; see README.md section "Third-party Notices"."""
    return P_aux_launched * fraction_of_external_power_coupled


@relation(
    name='External heating power',
    tags=('auxiliary_power',),
    outputs='P_external',
)
def external_heating_power(P_in: float, P_charged: float) -> Any:
    """Return external heating power from input power and power lost to charged particles.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    cfspopcon clips this at zero (near-ignition, P_in can sit numerically
    just below P_charged); fusdb's domain enforces P_external >= 0 exactly,
    so the clip is required rather than cosmetic.
    """
    return np.maximum(P_in - P_charged, 0.0)
