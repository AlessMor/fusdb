"""AMJUEL H.2 elastic scattering fits for H."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.1 H proton elastic total rate',
    tags=("atomic_physics",),
    outputs='H_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_1_h_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return H_proton_elastic_total_rate from AMJUEL H.2 subsection 2.1.
    
    Reaction: 0.1T p+H(1s) -> p+H(1s)totalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellian rate coefficient vs. Tp, with H(1s) at rest, obtained by taking the corresponding
    - Beam-Maxw. rate coefficient at Eb=0.08 eV and verification by independent integration of
    - cross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.1_H-proton-elastic-total", T_edge)


@relation(
    name='AMJUEL H.2 2.2 H proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='H_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_2_h_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return H_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.2.
    
    Reaction: 0.1D p+H(1s) -> p+H(1s)diff. rate coef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellian rate coefficient vs. Tp, with H(1s) at rest, obtained by taking the corresponding
    - Beam-Maxw. rate coefficient at Eb=0.08 eV and verification by independent integration of
    - cross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.2_H-proton-elastic-diffusion", T_edge)
