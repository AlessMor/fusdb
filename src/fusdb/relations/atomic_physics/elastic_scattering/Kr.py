"""AMJUEL H.2 elastic scattering fits for Kr."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.13 Kr proton elastic total rate',
    tags=("atomic_physics",),
    outputs='Kr_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_13_kr_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return Kr_proton_elastic_total_rate from AMJUEL H.2 subsection 2.13.
    
    Reaction: 0.7T p+Kr -> p+Kr totalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. T , with Ar at rest, obtained by taking the corresponding Beam-
    - p
    - Maxw. rate coefficient at Eb=0.05 eV and verification by independent integration of cross-
    - section"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_13.yaml", T_edge)


@relation(
    name='AMJUEL H.2 2.14 Kr proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='Kr_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_14_kr_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return Kr_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.14.
    
    Reaction: 0.7D p+Kr -> p+Krdiff. ratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. T ,withKratrest,obtainedbytakingthecorrespondingBeam-
    - p
    - Maxw. ratecoefficientatEb=0.1eVandverificationbyindependentintegrationofcross-section"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_14.yaml", T_edge)
