"""AMJUEL H.2 ionization fits for Ar."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.50 Ar ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Ar_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_50_ar_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Ar_ionization_rate from AMJUEL H.2 subsection 2.50.
    
    Reaction: 2.18B0 e+Ar -> e+Ar++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.1093 %, mean 0.0503 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization Rate for neutral Argon Atoms
    - Max. rel. Error: .1093 %
    - Mean rel. Error: .0503 %"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_50.yaml", T_edge)
