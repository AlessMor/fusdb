"""AMJUEL H.2 ionization fits for He."""

from typing import Any

from fusdb.relation import relation
from fusdb.utils.datasets import evaluate_amjuel_h2_rate

@relation(
    name='AMJUEL H.2 2.35 He ionization Freeman-Jones rate',
    tags=("atomic_physics",),
    outputs='He_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_35_he_ionization_freeman_minus_jones_rate(T_edge: Any) -> Any:
    """Return He_ionization_rate from AMJUEL H.2 subsection 2.35.
    
    Reaction: 2.2FJ e +He(1s21S) -> e+He+(1s)+e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Freeman and Jones rate coefficient for electron impact ionization of helium atoms [19] .
    - Data from impurity transport code "STRAHL" (K. Behringer) [17]
    - All reaction data with label ..aB0 or ..aB1 are taken from that reference. "a" is the
    - nuclear
    - charge number. aB0: ionisation of neutral atom. aB1: ionisation of singly charged ion.
    - Ionization Rate for neutral Helium Atoms
    - 2 +"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.35_He-ionization", T_edge)


@relation(
    name='AMJUEL H.2 2.36 He ionization STRAHL rate',
    tags=("atomic_physics",),
    outputs='He_ionization_rate',
    constraints=(
        'T_edge > 0.0',
    ),
)
def amjuel_h_2_2_36_he_ionization_strahl_rate(T_edge: Any) -> Any:
    """Return He_ionization_rate from AMJUEL H.2 subsection 2.36.
    
    Reaction: 2.2B0 e+He(1s 1S) -> e+He (1s)+e
    Input is T_edge [keV]; AMJUEL H.2 uses temperature in eV internally.
    The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 0.4138 %, mean 0.1636 %.
    
    Source: AMJUEL H.2 coefficient fit.
    AMJUEL comments:
    - Max. rel. Error: 0.4138 %
    - Mean rel. Error: 0.1636 %
    - Ionization Rate for single charged Helium Ions"""
    return evaluate_amjuel_h2_rate("polynomialfit_AMJUEL-H2-2.36_He-ionization", T_edge)
