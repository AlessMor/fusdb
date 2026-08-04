"""Species density and composition balance relations."""

from typing import Any

import numpy as np
from fusdb.utils import trapezoid

from fusdb.relation import relation
from ..utils import _positive_denominator


@relation(
    name="Ion density from tracked species densities",
    tags=("plasma", "composition"),
    outputs="n_i",
)
def ion_density_from_tracked_species_densities(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any = 0.0,
) -> Any:
    """Return total tracked ion density from species densities.

    ``n_p`` defaults to zero, matching how ``f_He3``/``f_He4`` are optional in
    the quasineutrality relations: a D-T case has no proton inventory, and
    making it positional would put ``n_p`` in the forward closure of every
    reactor and change which OTHER composition relations activate.
    """
    return n_D + n_T + n_He3 + n_He4 + n_p


@relation(
    name="D density from ion density and D fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_D",
)
def deuterium_density_from_ion_density_and_fraction(n_i: Any, f_D: Any) -> Any:
    """Return deuterium density from total ion density and D fraction."""
    return n_i * f_D


@relation(
    name="T density from ion density and T fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_T",
)
def tritium_density_from_ion_density_and_fraction(n_i: Any, f_T: Any) -> Any:
    """Return tritium density from total ion density and T fraction."""
    return n_i * f_T


@relation(
    name="He3 density from ion density and He3 fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_He3",
)
def helium3_density_from_ion_density_and_fraction(n_i: Any, f_He3: Any) -> Any:
    """Return helium-3 density from total ion density and He3 fraction."""
    return n_i * f_He3


@relation(
    name="He4 density from ion density and He4 fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_He4",
)
def helium4_density_from_ion_density_and_fraction(n_i: Any, f_He4: Any) -> Any:
    """Return helium-4 density from total ion density and He4 fraction."""
    return n_i * f_He4


@relation(
    name="p density from ion density and p fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_p",
)
def proton_density_from_ion_density_and_fraction(n_i: Any, f_p: Any) -> Any:
    """Return proton density from total ion density and proton fraction."""
    return n_i * f_p


def _reaction_balances(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return the pure D/T/He3/He4/p reaction source terms, without transport.

    Each term is ``stoichiometry * rate``, with the rate carrying a factor 1/2
    for like reactants.  The coefficients are transcribed by hand rather than
    generated, since this runs on the batched-popcon hot path; they are checked
    against ``registry.REACTIONS`` (reactions.yaml) by
    ``tests/test_reaction_stoichiometry.py``.

    Protons are a pure SOURCE: no tracked reaction consumes them (fusdb models
    no p-driven channel such as p-B11), so ``dn_p_dt`` has no negative term.
    """
    dn_D_dt = (
        -n_D * n_T * sigmav_DT
        - n_D**2 * (sigmav_DDn + sigmav_DDp)
        - n_D * n_He3 * sigmav_DHe3
        + n_T * n_He3 * sigmav_THe3_D
    )
    dn_T_dt = (
        +0.5 * n_D**2 * sigmav_DDp
        - n_D * n_T * sigmav_DT
        - n_T**2 * sigmav_TT
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
    )
    dn_He3_dt = (
        +0.5 * n_D**2 * sigmav_DDn
        - n_D * n_He3 * sigmav_DHe3
        - n_He3**2 * sigmav_He3He3
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
    )
    dn_He4_dt = (
        +n_D * n_T * sigmav_DT
        + n_D * n_He3 * sigmav_DHe3
        + 0.5 * n_T**2 * sigmav_TT
        + 0.5 * n_He3**2 * sigmav_He3He3
        + n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
    )
    dn_p_dt = (
        +0.5 * n_D**2 * sigmav_DDp
        + n_D * n_He3 * sigmav_DHe3
        # He3He3 -> p + p + He4: TWO protons per reaction, so the 1/2
        # like-reactant factor and the stoichiometry of 2 cancel.
        + n_He3**2 * sigmav_He3He3
        + n_T * n_He3 * sigmav_THe3_np
    )
    return dn_D_dt, dn_T_dt, dn_He3_dt, dn_He4_dt, dn_p_dt


def plasma_balance_ode(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p_D: Any,
    tau_p_T: Any,
    tau_p_He3: Any,
    tau_p_He4: Any,
    tau_p_p: Any,
    *,
    injection_fractions: np.ndarray | tuple[float, float, float, float, float] | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return D/T/He3/He4/p balances with implicit total-density fueling."""
    inv_tau_D = 0.0 if tau_p_D is None else 1.0 / tau_p_D
    inv_tau_T = 0.0 if tau_p_T is None else 1.0 / tau_p_T
    inv_tau_He3 = 0.0 if tau_p_He3 is None else 1.0 / tau_p_He3
    inv_tau_He4 = 0.0 if tau_p_He4 is None else 1.0 / tau_p_He4
    inv_tau_p = 0.0 if tau_p_p is None else 1.0 / tau_p_p

    src_D, src_T, src_He3, src_He4, src_p = _reaction_balances(
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
    )
    dn_D_dt = src_D - inv_tau_D * n_D
    dn_T_dt = src_T - inv_tau_T * n_T
    dn_He3_dt = src_He3 - inv_tau_He3 * n_He3
    dn_He4_dt = src_He4 - inv_tau_He4 * n_He4
    dn_p_dt = src_p - inv_tau_p * n_p

    total_density = n_D + n_T + n_He3 + n_He4 + n_p
    if injection_fractions is None:
        # Physical fuelling replenishes burned FUEL (D, T) in their current
        # ratio and never injects ash (He3/He4/p): injecting ash proportionally
        # to its own density is a positive feedback that drives the ash
        # fraction to unity instead of the trace steady state.  Protons are ash
        # here for the same reason helium is -- fusdb models no p-burning
        # channel, so a proton that enters the core only ever leaves by
        # transport.
        fuel_total_safe = np.maximum(n_D + n_T, 1e-300)
        zero = np.zeros_like(np.asarray(n_D, dtype=float))
        feed = np.stack(
            [n_D / fuel_total_safe, n_T / fuel_total_safe, zero, zero, zero], axis=0
        )
    else:
        feed = np.asarray(injection_fractions, dtype=float)
        if feed.shape[0] != 5 or not np.isfinite(feed).all() or np.any(feed < 0.0):
            raise ValueError("injection_fractions must be a length-5 non-negative vector")
        feed = feed / _positive_denominator(np.sum(feed, axis=0), name="injection fraction sum")

    net_balance = dn_D_dt + dn_T_dt + dn_He3_dt + dn_He4_dt + dn_p_dt
    source = -net_balance * feed
    return (
        dn_D_dt + source[0],
        dn_T_dt + source[1],
        dn_He3_dt + source[2],
        dn_He4_dt + source[3],
        dn_p_dt + source[4],
    )


def _normalized_balances(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return normalized particle balances for residual relations.

    A single global particle confinement time ``tau_p`` is applied to every
    species (uniform confinement); per-species loss could be reintroduced by
    splitting ``tau_p`` here.
    """
    balances = plasma_balance_ode(
        n_D,
        n_T,
        n_He3,
        n_He4,
        n_p,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p,
        tau_p,
        tau_p,
        tau_p,
        tau_p,
    )
    total_density = np.maximum(n_D + n_T + n_He3 + n_He4 + n_p, 1e-300)
    return tuple(balance / total_density for balance in balances)


def _integrated_balances(rho: Any, *balance_args: Any) -> tuple[float, float, float, float, float]:
    """Return the rho-integrated normalized particle balances (one scalar each).

    The steady-state balance is a relation between profiles, but the species
    are parameterized by scalar fractions (a profile of fixed shape ``n_i`` times
    a scalar ``f_X``).  The meaningful residual is therefore the profile reduced
    to its single free degree of freedom: the line-average over the ``rho`` grid,
    using the same trapezoid convention profiles use for their average.  This
    yields one scalar residual per species, matched to the scalar fraction it
    constrains, instead of an over-weighted bundle of per-point residuals that no
    scalar fraction can zero everywhere.
    """
    balances = _normalized_balances(*balance_args)
    rho_arr = np.asarray(rho, dtype=float).reshape(-1)
    if rho_arr.size <= 1:
        return tuple(float(np.asarray(b, dtype=float).reshape(-1)[0]) for b in balances)
    width = float(rho_arr[-1] - rho_arr[0]) or 1.0
    out: list[float] = []
    for b in balances:
        arr = np.asarray(b, dtype=float).reshape(-1)
        if arr.size != rho_arr.size:
            arr = np.full(rho_arr.size, float(arr[0]))
        out.append(float(trapezoid(arr, x=rho_arr) / width))
    return tuple(out)


@relation(name="Steady-state D particle balance", tags=("plasma", "composition"))
def steady_state_deuterium_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p: Any,
    rho: Any,
) -> Any:
    """Return rho-integrated D particle-balance residual."""
    return _integrated_balances(
        rho,
        n_D,
        n_T,
        n_He3,
        n_He4,
        n_p,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p,
    )[0]


@relation(name="Steady-state T particle balance", tags=("plasma", "composition"))
def steady_state_tritium_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p: Any,
    rho: Any,
) -> Any:
    """Return rho-integrated T particle-balance residual."""
    return _integrated_balances(
        rho,
        n_D,
        n_T,
        n_He3,
        n_He4,
        n_p,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p,
    )[1]


@relation(name="Steady-state He3 particle balance", tags=("plasma", "composition"))
def steady_state_helium3_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p: Any,
    rho: Any,
) -> Any:
    """Return rho-integrated He3 particle-balance residual."""
    return _integrated_balances(
        rho,
        n_D,
        n_T,
        n_He3,
        n_He4,
        n_p,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p,
    )[2]


@relation(name="Steady-state He4 particle balance", tags=("plasma", "composition"))
def steady_state_helium4_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p: Any,
    rho: Any,
) -> Any:
    """Return rho-integrated He4 particle-balance residual."""
    return _integrated_balances(
        rho,
        n_D,
        n_T,
        n_He3,
        n_He4,
        n_p,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p,
    )[3]


@relation(name="Steady-state p particle balance", tags=("plasma", "composition"))
def steady_state_proton_balance(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_p: Any,
    sigmav_DT: Any,
    sigmav_DDn: Any,
    sigmav_DDp: Any,
    sigmav_DHe3: Any,
    sigmav_TT: Any,
    sigmav_He3He3: Any,
    sigmav_THe3_D: Any,
    sigmav_THe3_np: Any,
    tau_p: Any,
    rho: Any,
) -> Any:
    """Return rho-integrated proton particle-balance residual."""
    return _integrated_balances(
        rho,
        n_D,
        n_T,
        n_He3,
        n_He4,
        n_p,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
        tau_p,
    )[4]
