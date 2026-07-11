"""AMJUEL H.2 ionization fits for C_plus."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.43 C+ ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='C_plus_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_43_c_plus_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return C_plus_ionization_rate from AMJUEL H.2 subsection 2.43.
    
    Reaction: 2.6B1 e+C+ -> e+C+++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.9478 %, mean 0.482 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization rate for Carbon Ions
    - Max. rel. Error: .9478 %
    - Mean rel. Error: .4820 %"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_43.yaml", T_edge)
