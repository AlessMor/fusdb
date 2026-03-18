"""Fusion reaction-rate relations using explicit profile integration."""

from __future__ import annotations

from fusdb.relation_util import relation
from fusdb.utils import integrate_profile_over_volume


################## TOTAL REACTION RATES ##################


def _is_symbolic(value: object) -> bool:
    """Return True for sympy-like symbolic values."""
    return bool(getattr(value, "free_symbols", None) is not None)


def _integrate_reaction_rate(integrand: float, V_p: float, *, label: str) -> float:
    """Integrate profile rates numerically, with symbolic fallback for model build."""
    if _is_symbolic(integrand) or _is_symbolic(V_p):
        # Symbolic model build uses scalar placeholders. For scalar profiles,
        # integrate_profile_over_volume(profile, V_p) == profile * V_p.
        return integrand * V_p

    total = integrate_profile_over_volume(integrand, V_p)
    if total is None:
        raise ValueError(f"Cannot integrate {label} reaction-rate profile over volume.")
    return total


@relation(
    name="DT reaction rate",
    output="Rr_DT",
    tags=("fusion_power",),
)
def reaction_rate_dt(f_D: float, f_T: float, n_i: float, sigmav_DT: float, V_p: float) -> float:
    """Return volume-integrated DT reaction rate [1/s]."""
    integrand = f_D * f_T * (n_i ** 2) * sigmav_DT
    return _integrate_reaction_rate(integrand, V_p, label="DT")


@relation(
    name="DD (He3+n) reaction rate",
    output="Rr_DDn",
    tags=("fusion_power",),
)
def reaction_rate_ddn(f_D: float, n_i: float, sigmav_DDn: float, V_p: float) -> float:
    """Return volume-integrated D(d,n)He3 reaction rate [1/s]."""
    integrand = 0.5 * (f_D ** 2) * (n_i ** 2) * sigmav_DDn
    return _integrate_reaction_rate(integrand, V_p, label="DDn")


@relation(
    name="DD (T+p) reaction rate",
    output="Rr_DDp",
    tags=("fusion_power",),
)
def reaction_rate_ddp(f_D: float, n_i: float, sigmav_DDp: float, V_p: float) -> float:
    """Return volume-integrated D(d,p)T reaction rate [1/s]."""
    integrand = 0.5 * (f_D ** 2) * (n_i ** 2) * sigmav_DDp
    return _integrate_reaction_rate(integrand, V_p, label="DDp")


@relation(
    name="D-He3 reaction rate",
    output="Rr_DHe3",
    tags=("fusion_power",),
)
def reaction_rate_dhe3(f_D: float, f_He3: float, n_i: float, sigmav_DHe3: float, V_p: float) -> float:
    """Return volume-integrated D-He3 reaction rate [1/s]."""
    integrand = f_D * f_He3 * (n_i ** 2) * sigmav_DHe3
    return _integrate_reaction_rate(integrand, V_p, label="DHe3")


@relation(
    name="T-T reaction rate",
    output="Rr_TT",
    tags=("fusion_power",),
)
def reaction_rate_tt(f_T: float, n_i: float, sigmav_TT: float, V_p: float) -> float:
    """Return volume-integrated T-T reaction rate [1/s]."""
    integrand = 0.5 * (f_T ** 2) * (n_i ** 2) * sigmav_TT
    return _integrate_reaction_rate(integrand, V_p, label="TT")


@relation(
    name="He3-He3 reaction rate",
    output="Rr_He3He3",
    tags=("fusion_power",),
)
def reaction_rate_he3he3(f_He3: float, n_i: float, sigmav_He3He3: float, V_p: float) -> float:
    """Return volume-integrated He3-He3 reaction rate [1/s]."""
    integrand = 0.5 * (f_He3 ** 2) * (n_i ** 2) * sigmav_He3He3
    return _integrate_reaction_rate(integrand, V_p, label="He3He3")


@relation(
    name="T-He3 alpha+D reaction rate",
    output="Rr_THe3_D",
    tags=("fusion_power",),
)
def reaction_rate_the3_d(f_T: float, f_He3: float, n_i: float, sigmav_THe3_D: float, V_p: float) -> float:
    """Return volume-integrated T-He3 alpha + D branch reaction rate [1/s]."""
    integrand = f_T * f_He3 * (n_i ** 2) * sigmav_THe3_D
    return _integrate_reaction_rate(integrand, V_p, label="THe3_D")


@relation(
    name="T-He3 alpha+n+p reaction rate",
    output="Rr_THe3_np",
    tags=("fusion_power",),
)
def reaction_rate_the3_np(f_T: float, f_He3: float, n_i: float, sigmav_THe3_np: float, V_p: float) -> float:
    """Return volume-integrated T-He3 alpha + n + p branch reaction rate [1/s]."""
    integrand = f_T * f_He3 * (n_i ** 2) * sigmav_THe3_np
    return _integrate_reaction_rate(integrand, V_p, label="THe3_np")


@relation(
    name="T-He3 reaction rate",
    output="Rr_THe3",
    tags=("fusion_power",),
)
def reaction_rate_the3(Rr_THe3_D: float, Rr_THe3_np: float) -> float:
    """Return total T-He3 reaction rate from the two implemented branches."""
    return Rr_THe3_D + Rr_THe3_np
