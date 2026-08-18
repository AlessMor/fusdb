"""AMJUEL H.2 molecular ionization fits for N2."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.29 N2 molecular ionization rate',
    tags=("atomic_physics",),
    outputs='N2_molecular_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_29_n2_molecular_ionization_rate(T_edge: Any) -> Any:
    """Return N2_molecular_ionization_rate from AMJUEL H.2 subsection 2.29.
    
    Reaction: 2.7.9 e + N -> e+N+ +e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - ionisation cross-section: total: [12], I, DI separate [13] (branching ratio R(E)) Here:
    - ionisation
    - to N+
    - DeltaEel = 15.581 eV KER=0."""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.29_N2-molecular-ionization", T_edge)
