"""AMJUEL H.2 charge exchange fits for H2."""

from pathlib import Path
from typing import Any

from fusdb import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h2_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.2 2.26 H2 charge exchange rate',
    tags=("atomic_physics",),
    outputs='H2_charge_exchange_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_26_h2_charge_exchange_rate(T_edge: Any) -> Any:
    """Return H2_charge_exchange_rate from AMJUEL H.2 subsection 2.26.
    
    Reaction: 3.2.3 p + H -> H +H+
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 10.2031 %, mean 6.3799 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - Effective ion conversion rate (charge exchange on H )
    - sum 2
    - <sigmav> =<sigmav> + 14 <sigmav> * pH (v)
    - eff H2(v=0) v=1 H2(v) 2
    - Same vibrational distribution (as function of Te) as above. Therefore: single parameter fit
    - vs. Te, since vibrational distribution does not depend upon density, E0 is fixed (0.1 eV)
    - and
    - T =T =T.
    - p e
    - Max. rel. Error: 10.2031 %
    - Mean rel. Error: 6.3799 %
    - Competing process at low T: see above: dissociative electron attachment, process 2-2-17"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_26.yaml", T_edge)


@relation(
    name='AMJUEL H.2 2.27 H2 charge exchange old rate',
    tags=("atomic_physics",),
    outputs='H2_charge_exchange_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_27_h2_charge_exchange_old_rate(T_edge: Any) -> Any:
    """Return H2_charge_exchange_rate from AMJUEL H.2 subsection 2.27.
    
    Reaction: 3.2.3o p + H -> H +H+
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 10.9657 %, mean 6.2957 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - Effective ion conversion rate (charge exchange on H (old version before 2004)
    - sum 2
    - <sigmav> =<sigmav> + 14 <sigmav> *pH (v)
    - eff H2(v=0) v=1 H2(v) 2
    - Same vibrational distribution (as function of Te) as above. Therefore: single parameter fit
    - vs. T , since vibrational distribution does not depend upon density, E is fixed (0.37 eV)
    - and
    - e 0
    - T =T =T.
    - p e
    - Max. rel. Error: 10.9657 %
    - Mean rel. Error: 6.2957 %
    - same as 3-2-3, above, previous reaction, but here evaluated with old default H energy: E =
    - 0.37 eV, rather than the current choice of E =0.1 eV. Old data are kept here only for
    - backward
    - compatibility. (The old rate coefficient is mostly used in ITER applications and SOLPS4.x in
    - general). Strictly this rate coefficient should be evaluated for stationary H (energy=0.0)
    - to
    - permit correct mass scaling in the Maxwellian averages.
    - Competing process at low T: see above: dissociative electron attachment, process 2-2-17
    - Next few reactions: rate coefficients, vs. Te, for a number of N ,N+ corona dissociation and
    - 2 2
    - ionisation channels"""
    return evaluate_amjuel_h2_rate(_DATA_DIR / "amjuel_h2_2_27.yaml", T_edge)
