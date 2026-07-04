"""AMJUEL H.2 elastic scattering fits for Ar."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.11 Ar proton elastic total rate',
    tags=("atomic_physics",),
    outputs='Ar_proton_elastic_total_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_11_ar_proton_elastic_total_rate(T_edge: Any) -> Any:
    """Return Ar_proton_elastic_total_rate from AMJUEL H.2 subsection 2.11.
    
    Reaction: 0.6T p+Ar -> p+Ar totalratecoef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. T , with Ar at rest, obtained by taking the corresponding Beam-
    - p
    - Maxw. rate coefficient at Eb=0.05 eV and verification by independent integration of cross-
    - section"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_11.yaml", T_edge)


@relation(
    name='AMJUEL H.2 2.12 Ar proton elastic diffusion rate',
    tags=("atomic_physics",),
    outputs='Ar_proton_elastic_diffusion_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_12_ar_proton_elastic_diffusion_rate(T_edge: Any) -> Any:
    """Return Ar_proton_elastic_diffusion_rate from AMJUEL H.2 subsection 2.12.
    
    Reaction: 0.6D p+Ar -> p+Ar diff. rate coef.
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Maxwellianratecoefficientvs. T , with Ar at rest, obtained by taking the corresponding Beam-
    - p
    - Maxw. rate coefficient at Eb=0.07 eV and verification by independent integration of cross-
    - section"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_12.yaml", T_edge)
