"""AMJUEL H.2 dissociation fits for N2."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.28 N2 dissociation rate',
    tags=("atomic_physics",),
    outputs='N2_dissociation_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_28_n2_dissociation_rate(T_edge: Any) -> Any:
    """Return N2_dissociation_rate from AMJUEL H.2 subsection 2.28.
    
    Reaction: 2.7.5 e + N -> e+N +N
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Dissociation from ground state N , cross-section from [10], DeltaE =9.7527eV,KER:0.95eV
    - 2 el
    - (spectra with two peaks, at 0.8 and 1.1 eV resp.)"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_28.yaml", T_edge)
