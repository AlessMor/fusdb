"""AMJUEL H.2 dissociative recombination fits for N2_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.33 N2+ dissociative recombination rate',
    tags=("atomic_physics",),
    outputs='N2_plus_dissociative_recombination_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_33_n2_plus_dissociative_recombination_rate(T_edge: Any) -> Any:
    """Return N2_plus_dissociative_recombination_rate from AMJUEL H.2 subsection 2.33.
    
    Reaction: 2.7.14 e + N+ -> N +N
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Dissociativerecombination,cross-section: [15]DeltaE canbetakenfromelectronenergyweighted
    - el
    - rate coefficient. KER: 1.06 - 5.824 eV at zero electron impact energy, depending on
    - vibrational
    - state of N+ and electronic state of products. Suggestion: KER = 3.5 eV"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.33_N2-plus-dissociative-recombination", T_edge)
