"""AMJUEL H.2 ionization fits for B."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.40 B ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='B_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_40_b_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return B_ionization_rate from AMJUEL H.2 subsection 2.40.
    
    Reaction: 2.5B0 e+B -> e+B++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization Rates for neutral Boron Atoms"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_40.yaml", T_edge)
