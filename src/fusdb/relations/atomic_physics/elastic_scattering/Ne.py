"""AMJUEL H.2 elastic scattering fits for Ne."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.9 Ne proton elastic total rate',
    tags=("atomic_physics",),
    outputs='Ne_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_9_ne_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return Ne_proton_elastic_total_rate from AMJUEL H.2 subsection 2.9.
    
    Reaction: 0.5T p+Ne -> p+Netotalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. T ,withNeatrest,obtainedbytakingthecorrespondingBeam-
    - p
    - Maxw. ratecoefficientatEb=0.2eVandverificationbyindependentintegrationofcross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.9_Ne-proton-elastic-total", T_edge)


@relation(
    name='AMJUEL H.2 2.10 Ne proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='Ne_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_10_ne_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return Ne_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.10.
    
    Reaction: 0.5D p+Ne -> p+Nediff. ratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. T ,withNeatrest,obtainedbytakingthecorrespondingBeam-
    - p
    - Maxw. ratecoefficientatEb=0.1eVandverificationbyindependentintegrationofcross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.10_Ne-proton-elastic-diffusion", T_edge)
