"""AMJUEL H.4 dissociative ionization fits for H2."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h4_rate

@relation(
    name='AMJUEL H.4 2.2.10 H2 dissociative ionization rate',
    tags=("atomic_physics",),
    outputs='H2_dissociative_ionization_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
        'T_edge >= 5e-05',
    ),
)
def amjuel_h4_2_2_10_h2_dissociative_ionization_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_dissociative_ionization_rate from AMJUEL H.4 subsection 4.12.

    Reaction: 2.2.10 e + H2 → 2e + H + H+
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 1.2041 %, mean 0.4804 %.

    Source: AMJUEL H.4 coefficient fit.
    """
    return evaluate_amjuel_h4_rate("polynomialfit_AMJUEL-H4-2.2.10_H2-dissociative-ionization", n_e_edge, T_edge)
