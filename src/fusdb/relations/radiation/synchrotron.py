"""Synchrotron radiation relations."""

from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from fusdb import relation


@relation(
    name="Synchrotron radiation",
    tags=("power_balance",),
    outputs="P_sync",
)
def calc_synchrotron_radiation(
    rho: Any,
    n_e: Any,
    T_e: Any,
    R: Any,
    a: Any,
    B0: Any,
    separatrix_elongation: Any,
    V_p: Any,
) -> Any:
    """Calculate the synchrotron radiated power due to the main plasma.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Formula 15 in :cite:`stott_feasibility_2005`. Assumes 90% wall reflectivity and
    profiles n(r)=n[1-(r/a)^2]^alpha_n, T(r)=Tedge+(T-Tedge)[1-(r/a)^gamma_T]^alpha_T
    with gamma_T=2, alpha_n=0.5, alpha_T=1.

    Note: cfspopcon volume-weights the profile integral (sum(f*2*rho*drho)*V);
    fusdb integrates uniformly in rho (V_p*trapezoid), matching the existing
    Bremsstrahlung relation -- the two radiated-power channels share that
    approximation.

    Args:
        rho: [~] :term:`glossary link<rho>`
        n_e: [1/m^3] :term:`glossary link<electron_density_profile>`
        T_e: [keV] :term:`glossary link<electron_temp_profile>`
        R: [m] :term:`glossary link<major_radius>`
        a: [m] :term:`glossary link<minor_radius>`
        B0: [T] :term:`glossary link<magnetic_field_on_axis>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        V_p: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
        Synchrotron radiated power [W]
    """
    # CHECK
    ne20 = n_e / 1e20

    Rw = 0.8  # wall reflectivity
    gamma_T = 2  # temperature profile inner exponent (2 is ~parabolic)
    alpha_n = 0.5  # density profile outer exponent (0.5 is rather broad)
    alpha_T = 1  # temperature profile outer exponent (1 is ~parabolic)

    # effective optical thickness
    rhoa = 6.04e3 * a * ne20 / B0
    # profile peaking correction
    Ks = (
        (alpha_n + 3.87 * alpha_T + 1.46) ** (-0.79)
        * (1.98 + alpha_n) ** (1.36)
        * gamma_T**2.14
        * (gamma_T**1.53 + 1.87 * alpha_T - 0.16) ** (-1.33)
    )
    # aspect ratio correction
    Gs = 0.93 * (1 + 0.85 * np.exp(-0.82 * R / a))

    # dimensionless parameter accounting for plasma transparency and wall reflections.
    # fusdb's parabolic profiles reach n_e=T_e=0 at the edge, where this term is a
    # 0*inf singularity; the edge synchrotron power density is physically zero, so
    # non-finite contributions are zeroed (cfspopcon avoids this via non-zero edge profiles).
    with np.errstate(divide="ignore", invalid="ignore"):
        Phi = (
            6.86e-5
            * separatrix_elongation ** (-0.21)
            * (16 + T_e) ** (2.61)
            * ((rhoa / (1 - Rw)) ** (0.41) + 0.12 * T_e) ** (-1.51)
            * Ks
            * Gs
        )
        p_sync = 6.25e-3 * ne20 * T_e * B0**2 * Phi * 1e6  # [W/m^3]
    p_sync = np.where(np.isfinite(p_sync), p_sync, 0.0)
    return V_p * trapezoid(p_sync, x=rho)
