"""AMJUEL H.2 dissociative ionization fits for N2_plus."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.31 N2+ dissociative ionization rate',
    tags=("atomic_physics",),
    outputs='N2_plus_dissociative_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_31_n2_plus_dissociative_ionization_rate(T_edge: Any) -> Any:
    """Return N2_plus_dissociative_ionization_rate from AMJUEL H.2 subsection 2.31.
    
    Reaction: 2.7.11 e + N+ -> e+2N+ +e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Dissociative ionisation, cross-section: [11] DeltaE =31.2eV,KER:max: 11.8eV
    - el"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_31.yaml", T_edge)
