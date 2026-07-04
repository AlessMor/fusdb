"""AMJUEL H.2 elastic scattering fits for H2."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.5 H2 proton elastic total rate',
    tags=("atomic_physics",),
    outputs='H2_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_5_h2_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return H2_proton_elastic_total_rate from AMJUEL H.2 subsection 2.5.
    
    Reaction: 0.3T p+H -> p+H totalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - Maxwellianratecoefficientvs. T , withH atrest, obtainedbytakingthecorrespondingBeam-
    - p 2
    - Maxw. rate coefficient at Eb=0.06 eV and verification by independent integration of cross-
    - section"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_5.yaml", T_edge)


@relation(
    name='AMJUEL H.2 2.6 H2 proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='H2_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_6_h2_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return H2_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.6.
    
    Reaction: 0.3D p+H -> p+H diff. ratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - Maxwellianratecoefficientvs. T , withH atrest, obtainedbytakingthecorrespondingBeam-
    - p 2
    - Maxw. ratecoefficientatEb=0.1eVandverificationbyindependentintegrationofcross-section
    - + 2 + 2"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_6.yaml", T_edge)
