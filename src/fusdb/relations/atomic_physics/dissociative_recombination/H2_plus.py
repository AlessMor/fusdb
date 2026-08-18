"""AMJUEL H.2 dissociative recombination fits for H2_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.22 H2+ dissociative recombination rate',
    tags=("atomic_physics",),
    outputs='H2_plus_dissociative_recombination_rate',
    constraints=(
        'T_edge > 0.0',
        'T_edge >= 0.0001',
    ),
)
def amjuel_h_2_2_22_h2_plus_dissociative_recombination_rate(T_edge: Any) -> Any:
    """Return H2_plus_dissociative_recombination_rate from AMJUEL H.2 subsection 2.22.
    
    Reaction: 2.2.14 e + H+(v) -> H(1s)+H(n)(v = 0...9,n >= 2)
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Fit as given in [2] but with all higher coefficients b2,b3,...b8 set to zero, for this
    - dissociative
    - recombination process. This latter fit seems to be more plausible. Therefore, the
    - (presumably
    - more correct) data are stored here, whereas the original data from ref.[2] are still given
    - in
    - HYDHEL,forreferencepurposesonly.
    - Tmin 1.00e-01 <sv>(Tmin) 2.23e-07 <sv>max 2.23e-07 Error 3.30e-13"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.22_H2-plus-dissociative-recombination", T_edge)
