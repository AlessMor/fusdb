"""Auxiliary power relations."""

from __future__ import annotations

from typing import Any

from fusdb import relation
@relation(
    name='Total auxiliary power',
    tags=('power_exhaust', 'auxiliary'),
    outputs='P_aux',
)
def auxiliary_power(P_NBI: float, P_ICRF: float, P_LHCD: float) -> Any:
    """Return total auxiliary power from injected sources."""
    return P_NBI + P_ICRF + P_LHCD
# TODO(med): specify launched and absorbed auxiliary power

# TODO(low): specify fraction of P_aux used for heating and for current drive

#from cfspopcon:
# def calc_auxiliary_power(P_in: Unitfull, P_alpha: Unitfull, P_ohmic: Unitfull, fraction_of_external_power_coupled: Unitfull) -> Unitfull:
#     """Calculate the required auxiliary power.

#     Args:
#         P_in: [MW] :term:`glossary link<P_in>`
#         P_alpha: [MW] :term:`glossary link<P_alpha>`
#         P_ohmic: [MW] :term:`glossary link<P_ohmic>`
#         fraction_of_external_power_coupled: [~]: :term:`glossary link<fraction_of_external_power_coupled>`

#     Returns:
#         :term:`P_external` [MW], :term:`P_auxiliary_absorbed` [MW], :term:`P_auxiliary_launched` [MW]
#     """
#     P_external = (P_in - P_alpha).clip(min=0.0 * ureg.MW)
#     P_auxiliary_absorbed = (P_external - P_ohmic).clip(min=0.0 * ureg.MW)
#     P_auxiliary_launched = P_auxiliary_absorbed / fraction_of_external_power_coupled

#     return P_external, P_auxiliary_absorbed, P_auxiliary_launched