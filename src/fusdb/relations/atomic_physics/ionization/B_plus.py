"""AMJUEL H.2 ionization fits for B_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.41 B+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='B_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_41_b_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return B_plus_ionization_rate from AMJUEL H.2 subsection 2.41.
    
    Reaction: 2.5B1 e+B+ -> e+B+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization Rates for single charged Boron Ions"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.41_B-plus-ionization", T_edge)
