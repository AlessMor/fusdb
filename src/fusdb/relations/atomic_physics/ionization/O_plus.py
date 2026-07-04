"""AMJUEL H.2 ionization fits for O_plus."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.47 O+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='O_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_47_o_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return O_plus_ionization_rate from AMJUEL H.2 subsection 2.47.
    
    Reaction: 2.8B1 e+O+ -> e+O+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization rate for singly charged Oxygen Ions
    - Ionization Rate for neutral Neon Atoms"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_47.yaml", T_edge)
