"""Fusion power relations using Bosch-Hale reactivities."""

from __future__ import annotations

from fusdb.reactor_class import Reactor
from fusdb.registry.constants import (
    DD_HE3_ENERGY_J,
    DD_N_ENERGY_J,
    DD_P_ENERGY_J,
    DD_T_ENERGY_J,
    DHE3_ALPHA_ENERGY_J,
    DHE3_P_ENERGY_J,
    DT_ALPHA_ENERGY_J,
    DT_N_ENERGY_J,
    TT_REACTION_ENERGY_J,
)
from fusdb.relations.fusion_power.reactivity_functions import (
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


@Reactor.relation(
    "fusion_power",
    name="DT alpha power",
    output="P_fus_DT_alpha",
    constraints=("P_fus_DT_alpha >= 0",),
)
def alpha_power_dt(Rr_DT: float) -> float:
    """Return alpha power from D-T fusion."""
    return DT_ALPHA_ENERGY_J * Rr_DT


@Reactor.relation(
    "fusion_power",
    name="DD triton power",
    output="P_fus_DDp_T",
    constraints=("P_fus_DDp_T >= 0",),
)
def triton_power_dd(Rr_DDp: float) -> float:
    """Return triton power from the D(d,p)T branch."""
    return DD_T_ENERGY_J * Rr_DDp


@Reactor.relation(
    "fusion_power",
    name="DD helium-3 power",
    output="P_fus_DDn_He3",
    constraints=("P_fus_DDn_He3 >= 0",),
)
def he3_power_dd(Rr_DDn: float) -> float:
    """Return He3 power from the D(d,n)He3 branch."""
    return DD_HE3_ENERGY_J * Rr_DDn


@Reactor.relation(
    "fusion_power",
    name="DD proton power",
    output="P_fus_DDp_p",
    constraints=("P_fus_DDp_p >= 0",),
)
def proton_power_dd(Rr_DDp: float) -> float:
    """Return proton power from the D(d,p)T branch."""
    return DD_P_ENERGY_J * Rr_DDp


@Reactor.relation(
    "fusion_power",
    name="D-He3 alpha power",
    output="P_fus_DHe3_alpha",
    constraints=("P_fus_DHe3_alpha >= 0",),
)
def alpha_power_dhe3(Rr_DHe3: float) -> float:
    """Return alpha power from D-He3 fusion."""
    return DHE3_ALPHA_ENERGY_J * Rr_DHe3


@Reactor.relation(
    "fusion_power",
    name="D-He3 proton power",
    output="P_fus_DHe3_p",
    constraints=("P_fus_DHe3_p >= 0",),
)
def proton_power_dhe3(Rr_DHe3: float) -> float:
    """Return proton power from D-He3 fusion."""
    return DHE3_P_ENERGY_J * Rr_DHe3


@Reactor.relation(
    "fusion_power",
    name="DT neutron power",
    output="P_fus_DT_n",
    constraints=("P_fus_DT_n >= 0",),
)
def neutron_power_dt(Rr_DT: float) -> float:
    """Return neutron power from D-T fusion."""
    return DT_N_ENERGY_J * Rr_DT


@Reactor.relation(
    "fusion_power",
    name="DD neutron power",
    output="P_fus_DDn_n",
    constraints=("P_fus_DDn_n >= 0",),
)
def neutron_power_dd(Rr_DDn: float) -> float:
    """Return neutron power from the D(d,n)He3 branch."""
    return DD_N_ENERGY_J * Rr_DDn


@Reactor.relation(
    "fusion_power",
    name="TT fusion power",
    output="P_fus_TT",
    constraints=("P_fus_TT >= 0",),
)
def fusion_power_tt(Rr_TT: float) -> float:
    """Return total fusion power from T-T reactions."""
    return TT_REACTION_ENERGY_J * Rr_TT


@Reactor.relation(
    "fusion_power",
    name="DT fusion power",
    output="P_fus_DT",
    constraints=("P_fus_DT >= 0",),
)
def fusion_power_dt(P_fus_DT_alpha: float, P_fus_DT_n: float) -> float:
    """Return total D-T fusion power from alpha and neutron components."""
    return P_fus_DT_alpha + P_fus_DT_n


@Reactor.relation(
    "fusion_power",
    name="DD (He3+n) fusion power",
    output="P_fus_DDn",
    constraints=("P_fus_DDn >= 0",),
)
def fusion_power_ddn(P_fus_DDn_He3: float, P_fus_DDn_n: float) -> float:
    """Return D-D fusion power from the He3+n branch."""
    return P_fus_DDn_He3 + P_fus_DDn_n


@Reactor.relation(
    "fusion_power",
    name="DD (T+p) fusion power",
    output="P_fus_DDp",
    constraints=("P_fus_DDp >= 0",),
)
def fusion_power_ddp(P_fus_DDp_T: float, P_fus_DDp_p: float) -> float:
    """Return D-D fusion power from the T+p branch."""
    return P_fus_DDp_T + P_fus_DDp_p


@Reactor.relation(
    "fusion_power",
    name="DD fusion power",
    output="P_fus_DD",
    constraints=("P_fus_DD >= 0",),
)
def fusion_power_dd(P_fus_DDn: float, P_fus_DDp: float) -> float:
    """Return total D-D fusion power from both branches."""
    return P_fus_DDn + P_fus_DDp


@Reactor.relation(
    "fusion_power",
    name="D-He3 fusion power",
    output="P_fus_DHe3",
    constraints=("P_fus_DHe3 >= 0",),
)
def fusion_power_dhe3(P_fus_DHe3_alpha: float, P_fus_DHe3_p: float) -> float:
    """Return total D-He3 fusion power from alpha and proton components."""
    return P_fus_DHe3_alpha + P_fus_DHe3_p


@Reactor.relation(
    "fusion_power",
    name="Total fusion power",
    output="P_fus",
    constraints=("P_fus >= 0",),
)
def fusion_power_total(
    P_fus_DT: float,
    P_fus_DD: float,
    P_fus_DHe3: float,
    P_fus_TT: float,
) -> float:
    """Return total fusion power from DT, DD, D-He3, and TT contributions."""
    return P_fus_DT + P_fus_DD + P_fus_DHe3 + P_fus_TT
