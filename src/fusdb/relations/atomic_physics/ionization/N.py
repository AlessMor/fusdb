"""AMJUEL H.2 ionization fits for N."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.44 N ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='N_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_44_n_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return N_ionization_rate from AMJUEL H.2 subsection 2.44.
    
    Reaction: 2.7B0 e+N -> e+N++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization rate coefficient for neutral Nitrogen Atoms (Bell et al., CLM-R216) [8]
    - <sigma*vrel>(Te)(cm**3/s),N -- > N+"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.44_N-ionization", T_edge)


@relation(
    name='AMJUEL H.2 2.45 N ionization Brook rate',
    tags=("atomic_physics",),
    outputs='N_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_45_n_ionization_brook_rate(T_edge: Any) -> Any:
    """Return N_ionization_rate from AMJUEL H.2 subsection 2.45.
    
    Reaction: 2.7 e + N -> e+N+ +e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - cross-section data from Brook, [14] for e + N- > N+ + 2e, same cross-section data source
    - was used for the Bell rate coefficient. (Checked, Oct.2013: original Bell report and Brook
    - cross-sections are identical)
    - DeltaEel = 14.5"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.45_N-ionization", T_edge)
