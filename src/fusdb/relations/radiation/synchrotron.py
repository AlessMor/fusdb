"""Synchrotron radiation relations."""

from typing import Any

import numpy as np
from fusdb.utils import volume_average

from fusdb.relation import relation


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
    w_V: Any = None,
) -> Any:
    """Calculate the synchrotron radiated power due to the main plasma.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Formula 15 in :cite:`stott_feasibility_2005`. Assumes 90% wall reflectivity and
    profiles n(r)=n[1-(r/a)^2]^alpha_n, T(r)=Tedge+(T-Tedge)[1-(r/a)^gamma_T]^alpha_T
    with gamma_T=2, alpha_n=0.5, alpha_T=1.

    Note: cfspopcon volume-weights the profile integral (sum(f*2*rho*drho)*V);
    fusdb uses the equivalent flux-volume weighting (V_p*volume_average),
    matching the Bremsstrahlung, impurity-line and fusion-rate relations.

    Args:
        rho: Common computational profile grid.
        n_e: [1/m^3] :term:`glossary link<electron_density_profile>`
        T_e: [keV] :term:`glossary link<electron_temp_profile>`
        R: [m] :term:`glossary link<major_radius>`
        a: [m] :term:`glossary link<minor_radius>`
        B0: [T] :term:`glossary link<magnetic_field_on_axis>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        V_p: [m^3] :term:`glossary link<plasma_volume>`
        w_V: Optional physical volume-integration weight on ``rho``.

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
    return V_p * volume_average(p_sync, rho, weight=w_V)


@relation(
    name="Synchrotron radiation Albajar-Fidone",
    tags=("power_balance", "process"),
    outputs="P_sync",
)
def psync_albajar_fidone(
    nd_plasma_electron_on_axis: Any,
    rminor: Any,
    b_plasma_toroidal_on_axis: Any,
    aspect: Any,
    alphan: Any,
    alphat: Any,
    tbeta: Any,
    temp_plasma_electron_on_axis_kev: Any,
    f_sync_reflect: Any,
    rmajor: Any,
    kappa: Any,
) -> Any:
    """Calculate the synchrotron radiation power (Albajar total-power fit with
    the Fidone geometry correction).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS returns this as a power density (dividing the Albajar total power
    by the plasma volume); fusdb's ``P_sync`` is the total power in W, so this
    port returns the total power (MW converted to W) and skips the division.

    Parameters
    ----------
    nd_plasma_electron_on_axis :
        Central electron density [m^-3]
    rminor :
        Plasma minor radius [m]
    b_plasma_toroidal_on_axis :
        Toroidal field on axis [T]
    aspect :
        Aspect ratio
    alphan :
        Density profile index
    alphat :
        Temperature profile index
    tbeta :
        Temperature profile inner index (rho**tbeta)
    temp_plasma_electron_on_axis_kev :
        Central electron temperature [keV]
    f_sync_reflect :
        Fraction of synchrotron radiation reflected by the wall
    rmajor :
        Plasma major radius [m]
    kappa :
        Plasma elongation

    Returns
    -------
    :
        Synchrotron radiated power [W]

    References
    ----------
        - F. Albajar, J. Johner, and G. Granata, "Improved calculation of synchrotron
          radiation losses in realistic tokamak plasmas," Nuclear Fusion, vol. 41,
          no. 6, pp. 665-678, Jun. 2001.
        - I. Fidone, G. Giruzzi, and G. Granata, "Synchrotron radiation loss in
          tokamaks of arbitrary geometry," Nuclear Fusion, vol. 41, no. 12,
          pp. 1755-1758, Dec. 2001.
    """
    # CHECK
    ne0_20 = 1.0e-20 * nd_plasma_electron_on_axis

    p_a0 = 6.04e3 * (rminor * ne0_20) / b_plasma_toroidal_on_axis

    g_function = 0.93 * (1.0 + 0.85 * np.exp(-0.82 * aspect))

    # TODO: PROCESS uses (1.98 + alphat)**1.36 here while cfspopcon's Ks (see
    # calc_synchrotron_radiation above) uses (1.98 + alpha_n)**1.36 for the same
    # Albajar K-factor -- one of the two codes deviates from the paper; copied
    # as-is from PROCESS, check against Albajar 2001.
    k_function = (
        (alphan + 3.87 * alphat + 1.46) ** -0.79
        * (1.98 + alphat) ** 1.36
        * tbeta**2.14
        * (tbeta**1.53 + 1.87 * alphat - 0.16) ** -1.33
    )

    dum = (
        1.0
        + 0.12
        * (temp_plasma_electron_on_axis_kev / p_a0**0.41)
        * (1.0 - f_sync_reflect) ** 0.41
    ) ** -1.51

    p_sync_mw = (
        3.84e-8
        * (1.0 - f_sync_reflect) ** 0.62
        * rmajor
        * rminor**1.38
        * kappa**0.79
        * b_plasma_toroidal_on_axis**2.62
        * ne0_20**0.38
        * temp_plasma_electron_on_axis_kev
        * (16.0 + temp_plasma_electron_on_axis_kev) ** 2.61
        * dum
        * g_function
        * k_function
    )

    return p_sync_mw * 1.0e6
