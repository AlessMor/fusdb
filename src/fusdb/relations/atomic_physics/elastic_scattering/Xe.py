"""AMJUEL H.2 elastic scattering fits for Xe."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.15 Xe proton elastic total rate',
    tags=("atomic_physics",),
    outputs='Xe_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_15_xe_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return Xe_proton_elastic_total_rate from AMJUEL H.2 subsection 2.15.
    
    Reaction: 0.8T p+Xe -> p+Xetotalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. Tp,withXeatrest,obtainedbytakingthecorrespondingBeam-
    - Maxw. ratecoefficientatEb=0.2eVandverificationbyindependentintegrationofcross-section"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.15_Xe-proton-elastic-total", T_edge)


@relation(
    name='AMJUEL H.2 2.16 Xe proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='Xe_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_16_xe_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return Xe_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.16.
    
    Reaction: 0.8D p+Xe -> p+Xediff. ratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. Tp,withXeatrest,obtainedbytakingthecorrespondingBeam-
    - Maxw. ratecoefficientatEb=0.1eVandverificationbyindependentintegrationofcross-section
    - Data from Freeman and Jones [19], for comparison with old cases.
    - Note: Maxwellian rate coefficients are taken at neutral particle energy = 0.0 eV vs.
    - temperature
    - (electron or ion temp., resp.) of the Maxwellian f . I.e. :
    -  maxw
    - <sigmav> = d v sigma(v )*v *f (v )
    - p p p maxw p
    - Theionimpactratescanbescaledtodifferentisotopesandtofiniteneutralparticletemperatures
    - T byevaluating the fits at an effective temperature T given by
    - n eff
    - T = MT + MT
    - eff M1 1 M2 2
    - Here M is the mass of the ion as used in the Freeman/Jones rate coefficients, M and M are
    - 1 2
    - the masses of the two isotopes in the particular collision process considered, and T and T
    - are
    - 1 2
    - the two temperatures."""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.16_Xe-proton-elastic-diffusion", T_edge)
