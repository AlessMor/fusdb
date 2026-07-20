"""AMJUEL H.2 dissociative attachment fits for H2."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.23 H2 dissociative attachment rate',
    tags=("atomic_physics",),
    outputs='H2_dissociative_attachment_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_23_h2_dissociative_attachment_rate(T_edge: Any) -> Any:
    """Return H2_dissociative_attachment_rate from AMJUEL H.2 subsection 2.23.
    
    Reaction: 2.2.17 e + H2 -> e+H2(v) -> H +H-
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 11.6159 %, mean 5.8452 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Effective dissociative attachment rate.
    - <sigmav> =<sigmav> +sum14 <sigmav> * pH (v)
    - eff H2(v=0) v=1 H2(v) 2
    - Vibrational distribution pH (v,T ) (vs. T ) taken into account. Only coupling to H (v) elec-
    - 2 e e 2
    - tronic ground state. No population of H (v) from electronically excited H*, no radiative
    - transi-
    - 2 2
    - tions between vibrational levels. Assume: incident H particle with 0.1 eV (for the rate
    - taken to
    - befor H atrest) and T = T , hence: density independent vibrational distribution and
    - effective
    - 2 i e
    - rate, as well as neutral molecule energy independent rate.
    - Competingprocesses: seeionconversion,below,andcontributiontodissociationviavibrational
    - states, i.e., enhanced transition into repulsive triplett 3b.... state.
    - Max. rel. Error: 11.6159 %
    - Mean rel. Error: 5.8452 %"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.23_H2-dissociative-attachment", T_edge)
