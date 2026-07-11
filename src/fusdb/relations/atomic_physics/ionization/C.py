"""AMJUEL H.2 ionization fits for C."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.42 C ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='C_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_42_c_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return C_ionization_rate from AMJUEL H.2 subsection 2.42.
    
    Reaction: 2.6B0 e+C -> e+C++e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.3712 %, mean 0.1458 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Ionization rate for neutral Carbon Atoms
    - <sigma*vrel>(Te)(cm**3/s),C -- > C+
    - Max. rel. Error: .3712 %
    - Mean rel. Error: .1458 %"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_42.yaml", T_edge)
