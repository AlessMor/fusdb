"""AMJUEL H.2 ionization fits for Be_plus."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.39 Be+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Be_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_39_be_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Be_plus_ionization_rate from AMJUEL H.2 subsection 2.39.
    
    Reaction: 2.4B1 e+Be+ -> e+Be+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.3962 %, mean 0.2225 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Max. rel. Error: .3962 %
    - Mean rel. Error: .2225 %"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_39.yaml", T_edge)
