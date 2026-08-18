"""AMJUEL H.2 dissociative ionization fits for N2."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.30 N2 dissociative ionization rate',
    tags=("atomic_physics",),
    outputs='N2_dissociative_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_30_n2_dissociative_ionization_rate(T_edge: Any) -> Any:
    """Return N2_dissociative_ionization_rate from AMJUEL H.2 subsection 2.30.
    
    Reaction: 2.7.10 e + N2 -> e+N +N+ +e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - ionisationcross-section: total: [12], I, DI separate [13] (branching ratio R(E)) Here:
    - dissociative
    - ionisation to N + N+
    - DeltaEel = 24.34 eV, KER: 8 eV (estimated, not clearly specified in paper)"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.30_N2-dissociative-ionization", T_edge)
