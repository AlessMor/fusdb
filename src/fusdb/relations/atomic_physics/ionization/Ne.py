"""AMJUEL H.2 ionization fits for Ne."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.48 Ne ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Ne_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_48_ne_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Ne_ionization_rate from AMJUEL H.2 subsection 2.48.
    
    Reaction: 2.10B0 e+Ne -> e+Ne++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.02 %, mean 0.0103 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Max. rel. Error: .0200 %
    - Mean rel. Error: .0103 %
    - Ionization Rate for single charged Neon Ions"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.48_Ne-ionization", T_edge)
