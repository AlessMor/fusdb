"""Auxiliary heating power relations."""

from typing import Any

from fusdb import relation


@relation(
    name='Total auxiliary power',
    tags=('auxiliary_power',),
    outputs='P_aux',
)
def auxiliary_power_from_sources(P_NBI: float, P_ICRF: float, P_LHCD: float) -> Any:
    """Return total auxiliary power from injected sources.
    # TODO: check if additional power sources should be included here (e.g. ECRH, EBW,...).
    """
    return P_NBI + P_ICRF + P_LHCD


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
    Adapted from cfspopcon; see README.md section "Third-party Notices"."""
    return P_in - P_charged