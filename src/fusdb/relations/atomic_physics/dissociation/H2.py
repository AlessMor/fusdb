"""AMJUEL H.2 dissociation fits for H2."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.21 H2 dissociation original rate',
    tags=("atomic_physics",),
    outputs='H2_dissociation_rate',
    constraints=(
        'T_edge > 0.0',
        'T_edge >= 0.00126',
    ),
)
def amjuel_h_2_2_21_h2_dissociation_original_rate(T_edge: Any) -> Any:
    """Return H2_dissociation_rate from AMJUEL H.2 subsection 2.21.
    
    Reaction: 2.2.5org e + H (X+S) -> ... -> e+H(1s)+H(1s)
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 g
    - Old fit as given in [2], probably based on incorrect cross-section data.
    - EIRENEuses,asdefault, the fit as given in preprint for [2], unless otherwise specified, for
    - this
    - dissociation process. This latter fit seems to be more plausible. Therefore, the (presumably
    - more correct) preprint data are stored in file HYDHEL, whereas the original data from
    - ref.[2]
    - are given here in AMJUEL, for reference purposes only.
    - + 3 + 3 + 3
    - e+H(X S)->e+H(b Sigma ,a Sigma ,c Pi )->e+H(1s)+H(1s)
    - 2 g u g u
    - Tmin 1.26e+00 <sv>(Tmin) 3.25e-12 <sv>max 3.82e-09 Error 1.07e-06"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.21_H2-dissociation", T_edge)


@relation(
    name='AMJUEL H.2 2.24 H2 dissociation via H- rate',
    tags=("atomic_physics",),
    outputs='H2_dissociation_via_h_minus_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_24_h2_dissociation_via_h_minus_rate(T_edge: Any) -> Any:
    """Return H2_dissociation_via_h_minus_rate from AMJUEL H.2 subsection 2.24.
    
    Reaction: 2.2.17s e + H -> H +H +e(DissviaH-,coldH )
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - 2 2
    - Effective (intermediate H- condensed) dissociation rate coefficient, via H- - - > H + H-
    - channel.
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
    - Competing processes: see H- MAR, H- MAD, below.
    - Max. rel. Error: 0.220E+02 %
    - Mean rel. Error: 0.113E+02 %"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.24_H2-dissociation-via-h-minus", T_edge)
