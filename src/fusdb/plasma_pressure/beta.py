"""Beta relations defined via Reactor.relation decorators."""

from __future__ import annotations

import sympy as sp

from fusdb.reactors_class import Reactor
from fusdb.relation_util import nonzero, symbol

MU0 = 4 * sp.pi * 1e-7


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


@Reactor.relation(
    "plasma",
    name="Normalized beta",
    output="beta_N",
    constraints=(nonzero(symbol("I_p"))),
)
def normalized_beta(beta_T: float, a: float, B0: float, I_p: float) -> float:
    """Wesson/Troyon normalization: beta_N = beta_T(%) * a * B0 / I_p."""
    # beta_T is a fraction; convert to % with factor 100 for standard beta_N definition.
    return beta_T * 100.0 * a * B0 / I_p


@Reactor.relation(
    "plasma",
    name="Troyon beta limit",
    output="beta_limit",
    constraints=(nonzero(symbol("a")), nonzero(symbol("B0")), nonzero(symbol("I_p"))),
)
def troyon_beta_limit(a: float, B0: float, I_p: float) -> float:
    """Approximate Troyon limit: beta (fraction) = 0.028 * I_p / (a * B0)."""
    return 0.028 * I_p / (a * B0)

__all__ = ["MU0"]
