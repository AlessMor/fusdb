"""AMJUEL H.4 recombination fits for H."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h4_rate

@relation(
    name='AMJUEL H.4 2.1.8 H recombination rate',
    tags=("atomic_physics",),
    outputs='H_recombination_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
        'T_edge >= 0.0001',
    ),
)
def amjuel_h4_2_1_8_h_recombination_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H_recombination_rate from AMJUEL H.4 subsection 4.6.

    Reaction: 2.1.8 H+ +e → H(1s)
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - Effective hydrogenic recombination rate Data: K. Sawada, T.Fujimoto, radiative + three-body
    - contribution, [7] June17: Fit range extended from 0.1 – 1e3 to 0.1 – 2e4
    """
    return evaluate_amjuel_h4_rate("polynomialfit_AMJUEL-H4-2.1.8_H-recombination", n_e_edge, T_edge)
