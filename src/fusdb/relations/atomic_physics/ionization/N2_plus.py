"""AMJUEL H.2 ionization fits for N2_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.34 N2+ ionization rate',
    tags=("atomic_physics",),
    outputs='N2_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_34_n2_plus_ionization_rate(T_edge: Any) -> Any:
    """Return N2_plus_ionization_rate from AMJUEL H.2 subsection 2.34.
    
    Reaction: 2.7.15 e + N+ -> e+N++ +e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - Single ionisation of N+, cross-section: [11]
    - DeltaEel = 27.12 eV, KER=0."""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.34_N2-plus-ionization", T_edge)
