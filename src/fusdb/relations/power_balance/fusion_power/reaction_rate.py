"""Fusion power relations using Bosch-Hale reactivities."""

from __future__ import annotations

from fusdb.reactor_class import Reactor
from fusdb.relations.power_balance.fusion_power.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3,
    sigmav_THe3,
    sigmav_TT,
)

################## REACTION RATES ##################

@Reactor.relation(
    "fusion_power",
    name="DT reaction rate",
    output="Rr_DT",
    solve_for=("Rr_DT",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_DT >= 0"),
)
def reaction_rate_dt(f_D: float, f_T: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return DT fusion reaction rate from fractions, temperature, and volume."""
    reactivity = sigmav_DT_BoschHale(T_avg)
    return f_D * f_T * (n_i ** 2) * reactivity * V_p


@Reactor.relation(
    "fusion_power",
    name="DD (He3+n) reaction rate",
    output="Rr_DDn",
    solve_for=("Rr_DDn",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_DDn >= 0"),
)
def reaction_rate_ddn(f_D: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return D(d,n)He3 reaction rate (branch) from fractions, temperature, and volume."""
    _, sigmav_ddn, _ = sigmav_DD_BoschHale(T_avg)
    return 0.5 * (f_D ** 2) * (n_i ** 2) * sigmav_ddn * V_p


@Reactor.relation(
    "fusion_power",
    name="DD (T+p) reaction rate",
    output="Rr_DDp",
    solve_for=("Rr_DDp",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_DDp >= 0"),
)
def reaction_rate_ddp(f_D: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return D(d,p)T reaction rate (branch) from fractions, temperature, and volume."""
    _, _, sigmav_ddp = sigmav_DD_BoschHale(T_avg)
    return 0.5 * (f_D ** 2) * (n_i ** 2) * sigmav_ddp * V_p


@Reactor.relation(
    "fusion_power",
    name="D-He3 reaction rate",
    output="Rr_DHe3",
    solve_for=("Rr_DHe3",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_DHe3 >= 0"),
)
def reaction_rate_dhe3(f_D: float, f_He3: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return D-He3 fusion reaction rate from fractions, temperature, and volume."""
    reactivity = sigmav_DHe3_BoschHale(T_avg)
    return f_D * f_He3 * (n_i ** 2) * reactivity * V_p


@Reactor.relation(
    "fusion_power",
    name="T-T reaction rate",
    output="Rr_TT",
    solve_for=("Rr_TT",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_TT >= 0"),
)
def reaction_rate_tt(f_T: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return T-T fusion reaction rate from fractions, temperature, and volume."""
    reactivity = sigmav_TT(T_avg)
    return 0.5 * (f_T ** 2) * (n_i ** 2) * reactivity * V_p


@Reactor.relation(
    "fusion_power",
    name="He3-He3 reaction rate",
    output="Rr_He3He3",
    solve_for=("Rr_He3He3",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_He3He3 >= 0"),
)
def reaction_rate_he3he3(f_He3: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return He3-He3 fusion reaction rate from fractions, temperature, and volume."""
    reactivity = sigmav_He3He3(T_avg)
    return 0.5 * (f_He3 ** 2) * (n_i ** 2) * reactivity * V_p


@Reactor.relation(
    "fusion_power",
    name="T-He3 reaction rate",
    output="Rr_THe3",
    solve_for=("Rr_THe3",),
    constraints=("T_avg >= 0", "V_p >= 0", "Rr_THe3 >= 0"),
)
def reaction_rate_the3(f_T: float, f_He3: float, n_i: float, T_avg: float, V_p: float) -> float:
    """Return T-He3 fusion reaction rate from fractions, temperature, and volume."""
    reactivity = sigmav_THe3(T_avg)
    return f_T * f_He3 * (n_i ** 2) * reactivity * V_p
