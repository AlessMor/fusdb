"""AMJUEL H.2 ionization fits for Ne_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.49 Ne+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Ne_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_49_ne_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Ne_plus_ionization_rate from AMJUEL H.2 subsection 2.49.
    
    Reaction: 2.10B1 e+Ne+ -> e+Ne+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.1916 %, mean 0.0814 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Max. rel. Error: .1916 %
    - Mean rel. Error: .0814 %"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.49_Ne-plus-ionization", T_edge)
