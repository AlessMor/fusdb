"""AMJUEL H.2 ionization fits for He_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.37 He+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='He_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_37_he_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return He_plus_ionization_rate from AMJUEL H.2 subsection 2.37.
    
    Reaction: 2.2B1 e+He+(1s) -> e+He+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.9472 %, mean 0.5457 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Max. rel. Error: 0.9472 %
    - Mean rel. Error: 0.5457 %
    - Ionization Rates for neutral Beryllium Atoms"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.37_He-plus-ionization", T_edge)
