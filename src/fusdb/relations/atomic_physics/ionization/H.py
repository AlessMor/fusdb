"""AMJUEL H.2 ionization fits for H."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.17 H ionization Freeman-Jones rate',
    tags=("atomic_physics",),
    outputs='H_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_17_h_ionization_freeman_minus_jones_rate(T_edge: Any) -> Any:
    """Return H_ionization_rate from AMJUEL H.2 subsection 2.17.
    
    Reaction: 2.1.5FJ e + H(1s) -> e+H+ +e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit."""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_17.yaml", T_edge)


@relation(
    name='AMJUEL H.2 2.18 H proton-impact ionization Freeman-Jones rate',
    tags=("atomic_physics",),
    outputs='H_proton_impact_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_18_h_proton_minus_impact_ionization_freeman_minus_jones_rate(T_edge: Any) -> Any:
    """Return H_proton_impact_ionization_rate from AMJUEL H.2 subsection 2.18.
    
    Reaction: 3.1.6FJ p + H(1s) -> p+p+e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - This fit seems to be completely corrupted. Probably misprints in original F.J. CLM-R-137
    - report. Checked also with old AURORA code (PPPL, ca. 1979) implementation. Identical
    - fit used there. Recommendation: Use cross-section and HYDKIN online integration to rate
    - coefficients."""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_18.yaml", T_edge)
