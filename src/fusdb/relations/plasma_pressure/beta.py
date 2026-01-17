"""Beta relations defined via Reactor.relation decorators."""

from __future__ import annotations

import sympy as sp

from fusdb.registry.constants import MU0
from fusdb.reactor_class import Reactor
from fusdb.relation_util import nonzero, symbol

#TODO(low): once profiles are implemented, add a beta_avg 
@Reactor.relation(
    "plasma",
    name="Toroidal beta",
    output="beta_T",
    constraints=(nonzero(symbol("B0")),),
)
def toroidal_beta(p_th: float, B0: float) -> float:
    """Freidberg Eq. 11.58: toroidal field contribution (dimensionless)."""
    return (2 * MU0 * p_th) / (B0 ** 2)


@Reactor.relation(
    "plasma",
    name="Poloidal beta",
    output="beta_p",
    constraints=(nonzero(symbol("B_p")),),
)
def poloidal_beta(p_th: float, B_p: float) -> float:
    """Freidberg Eq. 11.58 using poloidal field magnitude."""
    return (2 * MU0 * p_th) / (B_p ** 2)


@Reactor.relation(
    "plasma",
    name="Beta decomposition",
    output="beta",
    constraints=(nonzero(symbol("beta_T")), nonzero(symbol("beta_p"))),
)
def beta_decomposition(beta_T: float, beta_p: float) -> float:
    """Total beta from toroidal and poloidal components (Freidberg Eq. 11.59)."""
    return 1 / (1 / beta_T + 1 / beta_p)
# TODO(high): call it also beta_total?


@Reactor.relation(
    "plasma",
    name="Normalized beta",
    output="beta_N",
    constraints=(nonzero(symbol("I_p"))),
)
def normalized_beta(beta_T: float, a: float, B0: float, I_p: float) -> float:
    """Wesson/Troyon normalization: beta_N = beta_T(%) * a * B0 / I_p."""
    # beta_T is a fraction; convert to % with factor 100 for standard beta_N definition.
    I_p_MA = I_p / 1e6
    return beta_T * 100.0 * a * B0 / I_p_MA
