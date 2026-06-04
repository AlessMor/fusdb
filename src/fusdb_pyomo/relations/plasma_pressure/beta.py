"""Beta relations defined via @relation decorators."""

from __future__ import annotations

from typing import Any

from fusdb_pyomo import relation
from fusdb_pyomo.registry import MU0
#TODO(low): once profiles are implemented, add a beta_avg 
@relation(
    name='Toroidal beta',
    tags=('plasma',),
    
    outputs='beta_T',
)
def toroidal_beta(p_th: float, B0: float) -> Any:
    """Freidberg Eq. 11.58: toroidal field contribution (dimensionless)."""
    return (2 * MU0 * p_th) / (B0 ** 2)


########################################
@relation(
    name='Poloidal beta',
    tags=('plasma',),
    
    outputs='beta_p',
)
def poloidal_beta(p_th: float, B_p: float) -> Any:
    """Freidberg Eq. 11.58 using poloidal field magnitude."""
    return (2 * MU0 * p_th) / (B_p ** 2)


########################################
@relation(
    name='Beta decomposition',
    tags=('plasma',),
    
    outputs='beta',
)
def beta_decomposition(beta_T: float, beta_p: float) -> Any:
    """Total beta from toroidal and poloidal components (Freidberg Eq. 11.59)."""
    return 1 / (1 / beta_T + 1 / beta_p)
# TODO(high): call it also beta_total?


########################################
@relation(
    name='Normalized beta',
    tags=('plasma',),
    
    outputs='beta_N',
)
def normalized_beta(beta_T: float, a: float, B0: float, I_p: float) -> Any:
    """Wesson/Troyon normalization: beta_N = beta_T(%) * a * B0 / I_p."""
    # beta_T is a fraction; convert to % with factor 100 for standard beta_N definition.
    I_p_MA = I_p / 1e6
    return beta_T * 100.0 * a * B0 / I_p_MA
