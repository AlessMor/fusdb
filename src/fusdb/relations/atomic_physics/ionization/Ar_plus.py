"""AMJUEL H.2 ionization fits for Ar_plus."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.51 Ar+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Ar_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_51_ar_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Ar_plus_ionization_rate from AMJUEL H.2 subsection 2.51.
    
    Reaction: 2.18B1 e+Ar+ -> e+Ar+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.3659 %, mean 0.2214 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization Rate for single charged Argon Ions
    - Max. rel. Error: .3659 %
    - Mean rel. Error: .2214 %"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_51.yaml", T_edge)
