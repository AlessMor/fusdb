"""AMJUEL H.4 molecular ionization fits for H2."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h4_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.4 2.2.9 H2 molecular ionization rate',
    tags=("atomic_physics",),
    outputs='H2_molecular_ionization_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
    ),
)
def amjuel_h4_2_2_9_h2_molecular_ionization_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_molecular_ionization_rate from AMJUEL H.4 subsection 4.11.

    Reaction: 2.2.9 e + H2 → 2e + H2+
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 3.1001 %, mean 0.474 %.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - 2
    """
    return evaluate_amjuel_h4_rate(_DATA_DIR / "amjuel_h4_2_2_9.yaml", n_e_edge, T_edge)
