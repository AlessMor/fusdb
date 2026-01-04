"""Beta relations defined via Reactor.relation decorators."""

from __future__ import annotations

import math

from fusdb.reactors_class import Reactor
from fusdb.relations_util import require_nonzero

MU0 = 4 * math.pi * 1e-7


@Reactor.relation("plasma", name="Toroidal beta", output="beta_T", solve_for=("beta_T", "p_th", "B0"))
def toroidal_beta(p_th: float, B0: float) -> float:
    """Freidberg Eq. 11.58: toroidal field contribution (dimensionless)."""
    require_nonzero(B0, "B0", "toroidal beta")
    return (2 * MU0 * p_th) / (B0 ** 2)


@Reactor.relation("plasma", name="Poloidal beta", output="beta_p", solve_for=("beta_p", "p_th", "B_p"))
def poloidal_beta(p_th: float, B_p: float) -> float:
    """Freidberg Eq. 11.58 using poloidal field magnitude."""
    require_nonzero(B_p, "B_p", "poloidal beta")
    return (2 * MU0 * p_th) / (B_p ** 2)


@Reactor.relation("plasma", name="Beta decomposition", output="beta", solve_for=("beta", "beta_T", "beta_p"))
def beta_decomposition(beta_T: float, beta_p: float) -> float:
    """Total beta from toroidal and poloidal components (Freidberg Eq. 11.59)."""
    require_nonzero(beta_T, "beta_T", "beta decomposition")
    require_nonzero(beta_p, "beta_p", "beta decomposition")
    return 1 / (1 / beta_T + 1 / beta_p)


@Reactor.relation(
    "plasma",
    name="Normalized beta",
    output="beta_N",
    solve_for=("beta_N", "beta_T"),
)
def normalized_beta(beta_T: float, a: float, B0: float, I_p: float) -> float:
    """Wesson/Troyon normalization: beta_N = beta_T(%) * a * B0 / I_p."""
    require_nonzero(a, "a", "normalized beta")
    require_nonzero(B0, "B0", "normalized beta")
    require_nonzero(I_p, "I_p", "normalized beta")
    # beta_T is a fraction; convert to % with factor 100 for standard beta_N definition.
    return beta_T * 100.0 * a * B0 / I_p


@Reactor.relation(
    "plasma",
    name="Troyon beta limit",
    output="beta_limit",
    solve_for=("beta_limit",),
)
def troyon_beta_limit(a: float, B0: float, I_p: float) -> float:
    """Approximate Troyon limit: beta (fraction) = 0.028 * I_p / (a * B0)."""
    require_nonzero(a, "a", "Troyon limit")
    require_nonzero(B0, "B0", "Troyon limit")
    require_nonzero(I_p, "I_p", "Troyon limit")
    return 0.028 * I_p / (a * B0)

__all__ = ["MU0"]
