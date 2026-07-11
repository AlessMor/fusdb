"""AMJUEL H.2 ionization fits for Fe."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.52 Fe ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='Fe_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_52_fe_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return Fe_ionization_rate from AMJUEL H.2 subsection 2.52.
    
    Reaction: 2.26B0 e+Fe -> e+Fe++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.0907 %, mean 0.045 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization Rate for neutral Iron Atoms
    - Max. rel. Error: .0907 %
    - Mean rel. Error: .0450 %"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_52.yaml", T_edge)
