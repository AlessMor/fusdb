"""AMJUEL H.2 charge exchange fits for H."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.19 H charge exchange rate',
    tags=("atomic_physics",),
    outputs='H_charge_exchange_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_19_h_charge_exchange_rate(T_edge: Any) -> Any:
    """Return H_charge_exchange_rate from AMJUEL H.2 subsection 2.19.
    
    Reaction: 3.1.8 p + H(1s) -> H(1s)+p
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - addedbyDR:singleparameterMaxwellianratecoeff.,vs. T ,forneutraltargetatrest,obtained
    - p
    - from corresponding fit for Beam-Maxwellian rate coeff. evaluated at E = 0.1 eV and then
    - b
    - verified by independent integration of cross-section with proper low energy asymptotics."""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.19_H-charge-exchange", T_edge)


@relation(
    name='AMJUEL H.2 2.20 H charge exchange Freeman-Jones rate',
    tags=("atomic_physics",),
    outputs='H_charge_exchange_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_20_h_charge_exchange_freeman_minus_jones_rate(T_edge: Any) -> Any:
    """Return H_charge_exchange_rate from AMJUEL H.2 subsection 2.20.
    
    Reaction: 3.1.8FJ p + H(1s) -> H(1s)+p
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit."""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.20_H-charge-exchange", T_edge)


@relation(
    name='AMJUEL H.2 2.25 H charge exchange Langevin rate',
    tags=("atomic_physics",),
    outputs='H_charge_exchange_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_25_h_charge_exchange_langevin_rate(T_edge: Any) -> Any:
    """Return H_charge_exchange_rate from AMJUEL H.2 subsection 2.25.
    
    Reaction: 3.1.8L p + H(1s) -> H(1s)+p
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Langevin rate coefficient, constant at 2e-8"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.25_H-charge-exchange", T_edge)
