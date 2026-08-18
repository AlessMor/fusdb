"""AMJUEL H.2 ionization fits for Fe_plus."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.53 Fe+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Fe_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_53_fe_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Fe_plus_ionization_rate from AMJUEL H.2 subsection 2.53.
    
    Reaction: 2.26B1 e+Fe+ -> e+Fe+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.2106 %, mean 0.1105 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization Rate for single charged Iron Ions
    - Max. rel. Error: .2106 %
    - Mean rel. Error: .1105 %"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.53_Fe-plus-ionization", T_edge)
