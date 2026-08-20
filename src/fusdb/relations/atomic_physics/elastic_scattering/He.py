"""AMJUEL H.2 elastic scattering fits for He."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.3 He proton elastic total rate',
    tags=("atomic_physics",),
    outputs='He_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_3_he_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return He_proton_elastic_total_rate from AMJUEL H.2 subsection 2.3.
    
    Reaction: 0.2T p+He(1s21S) -> p+He(1s21S)totalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellian rate coefficient vs. T , with He(1s 1S) at rest, obtained by taking the
    - correspond-
    - p
    - ing Beam-Maxw. rate coefficient at Eb=0.2 eV and verification by independent integration of
    - cross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.3_He-proton-elastic-total", T_edge)


@relation(
    name='AMJUEL H.2 2.4 He proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='He_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_4_he_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return He_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.4.
    
    Reaction: 0.2D p+He(1s21S) -> p+He(1s21S)diff. rate coef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellian rate coefficient vs. T , with He(1s 1S) at rest, obtained by taking the
    - correspond-
    - p
    - ing Beam-Maxw. rate coefficient at Eb=0.2 eV and verification by independent integration of
    - cross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.4_He-proton-elastic-diffusion", T_edge)


@relation(
    name='AMJUEL H.2 2.7 He self elastic total rate',
    tags=("atomic_physics",),
    outputs='He_self_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_7_he_self_elastic_total_rate(T_edge: Any) -> Any:
    """Return He_self_elastic_total_rate from AMJUEL H.2 subsection 2.7.
    
    Reaction: 0.4T He (1s) + He(1s 1S) -> He (1s)+He(1s 1S) total
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - rate coef.
    - Maxwellian rate coefficient vs. T +, with He at rest, obtained by taking the corresponding
    - He
    - Beam-Maxw. ratecoefficientatEb=0.2eVandverificationbyindependentintegrationofcross-
    - section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.7_He-self-elastic-total", T_edge)


@relation(
    name='AMJUEL H.2 2.8 He self elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='He_self_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_8_he_self_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return He_self_elastic_diffusion_rate from AMJUEL H.2 subsection 2.8.
    
    Reaction: 0.4D He+(1s) + He(1s21S) -> He+(1s) + He(1s21S) diff.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - rate coef.
    - Maxwellian rate coefficient vs. T +, with He at rest, obtained by taking the corresponding
    - He
    - Beam-Maxw. ratecoefficientatEb=0.2eVandverificationbyindependentintegrationofcross-
    - section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.8_He-self-elastic-diffusion", T_edge)
