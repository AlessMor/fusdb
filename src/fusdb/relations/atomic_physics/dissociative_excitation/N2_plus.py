"""AMJUEL H.2 dissociative excitation fits for N2_plus."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.32 N2+ dissociative excitation rate',
    tags=("atomic_physics",),
    outputs='N2_plus_dissociative_excitation_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_32_n2_plus_dissociative_excitation_rate(T_edge: Any) -> Any:
    """Return N2_plus_dissociative_excitation_rate from AMJUEL H.2 subsection 2.32.
    
    Reaction: 2.7.12 e + N+ -> e+N +N+
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Dissociative excitation, cross-section: [11] DeltaE = 8.4 eV, KER: max. of 6.4 eV at 120 eV,
    - el
    - KERnearly=0nearthreshold(i.e. pre-dissociation via various channels)."""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_32.yaml", T_edge)
