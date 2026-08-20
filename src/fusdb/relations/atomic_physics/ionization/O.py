"""AMJUEL H.2 ionization fits for O."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.46 O ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='O_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_46_o_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return O_ionization_rate from AMJUEL H.2 subsection 2.46.
    
    Reaction: 2.8B0 e+O -> e+O++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization rate for neutral Oxygen Atoms
    - <sigma*vrel>(Te)(cm**3/s),O-->O+"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.46_O-ionization", T_edge)
