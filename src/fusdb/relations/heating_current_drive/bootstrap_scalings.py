"""Bootstrap-current-fraction scalings from PROCESS.

Ported from PROCESS ``process/models/physics/bootstrap_current.py``
(``PlasmaBootstrapCurrent``). The ``i_bootstrap_current`` enum dispatcher is
split into one relation per scaling; all output ``f_BS`` and are gated in
variables.yaml so fusdb's existing (cfspopcon Gi) bootstrap stays the default.

PROCESS multiplies every scaling by the user tuning coefficient ``cboot``
(default 1.0); that multiplier is orchestration and is not ported.

Not imported / skipped (logged in the import report):
    * gi_I -- mathematically identical to fusdb's existing "Bootstrap current
      fraction" (cfspopcon Gi 2014, scaling 1); PROCESS parameterises by
      profile indices + q95/q0 where fusdb uses peaking factors + qstar (q0=1).
    * wilson -- numerically derives profile indices from an assumed q-profile
      and raises on illegal values; needs a thermal-poloidal-beta variable.
    * nevins -- scipy.quad flux-surface integral over on-axis beta_e.
    * sugiyama_h_mode -- needs pedestal variables (pedestal radius/density/
      temperature, Greenwald density) fusdb does not model.
    * SauterBootstrapCurrent -- detailed per-flux-surface Sauter model over a
      profile object (collisionality helpers); an orchestrator.
"""

import numpy as np
from scipy import integrate
from scipy.special import beta as _beta_fn

from fusdb import relation
from fusdb.registry import ELECTRON_CHARGE_C, MU0

_TAGS = ("plasma", "current_drive", "tokamak", "process")


@relation(name="Bootstrap fraction ITER-89", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_iter89(
    A: float, beta: float, B_total: float, I_p: float, q95: float, q0: float, R: float, V_p: float
) -> float:
    """Bootstrap current fraction, original ITER (IPDG89) calculation.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    c_bs = 1.32 - 0.235 * (q95 / q0) + 0.0185 * (q95 / q0) ** 2
    average_a = np.sqrt(V_p / (2 * np.pi**2 * R))
    b_pa = (I_p / 1e6) / (5 * average_a)
    betapbs = beta * (B_total / b_pa) ** 2
    if betapbs <= 0.0:
        return 0.0
    return c_bs * (betapbs / np.sqrt(A)) ** 1.3


@relation(name="Bootstrap fraction Sakai", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_sakai(
    beta_p: float, q95: float, q0: float, alphan: float, alphat: float,
    eps: float, internal_inductivity: float,
) -> float:
    """Bootstrap current fraction, Sakai et al scaling (includes the diamagnetic
    fraction; use i_diamagnetic_current = 0 with this).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        10 ** (0.951 * eps - 0.948)
        * beta_p ** (1.226 * eps + 1.584)
        * internal_inductivity ** (-0.184 * eps - 0.282)
        * (q95 / q0) ** (-0.042 * eps - 0.02)
        * alphan ** (0.13 * eps + 0.05)
        * alphat ** (0.502 * eps - 0.273)
    )


@relation(name="Bootstrap fraction ARIES", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_aries(
    beta_p: float, internal_inductivity: float, n0: float, n_e_avg: float, eps: float
) -> float:
    """Bootstrap current fraction, ARIES scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    a_1 = 1.10 - 1.165 * internal_inductivity + 0.47 * internal_inductivity**2
    b_1 = 0.806 - 0.885 * internal_inductivity + 0.297 * internal_inductivity**2
    c_bs = a_1 + b_1 * (n0 / n_e_avg)
    return c_bs * np.sqrt(eps) * beta_p


@relation(name="Bootstrap fraction Andrade", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_andrade(
    beta_p: float, pres_plasma_on_axis: float, p_th: float, eps: float
) -> float:
    """Bootstrap current fraction, Andrade et al scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``pres_plasma_on_axis`` is the core pressure, ``p_th`` the volume-averaged
    thermal pressure.
    """
    # CHECK
    c_p = pres_plasma_on_axis / p_th
    c_bs = 0.2340
    return c_bs * np.sqrt(eps) * beta_p * c_p**0.8


@relation(name="Bootstrap fraction Hoang", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_hoang(beta_p: float, alphap: float, alphaj: float, eps: float) -> float:
    """Bootstrap current fraction, Hoang et al scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Uses (profile_index + 1) as the core/volume-averaged ratio, assuming
    parabolic pressure/current profiles.
    """
    # CHECK
    c_bs = np.sqrt((alphap + 1) / (alphaj + 1))
    return 0.4 * np.sqrt(eps) * beta_p**0.9 * c_bs


@relation(name="Bootstrap fraction Wong", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_wong(
    beta_p: float, alphan: float, alphat: float, eps: float, kappa: float
) -> float:
    """Bootstrap current fraction, Wong et al scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    f_peak = 2.0 / _beta_fn(0.5, alphan + alphat + 1)
    c_bs = 0.773 + 0.019 * kappa
    return c_bs * f_peak**0.25 * beta_p * np.sqrt(eps)


@relation(name="Bootstrap fraction Gi-II", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_gi_ii(
    beta_p: float, alphap: float, alphat: float, eps: float, Z_eff: float
) -> float:
    """Bootstrap current fraction, Gi et al scaling 2 (q-profile dependence
    removed relative to scaling 1).

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    c_bs = (
        0.382
        * eps**-0.242
        * alphap**0.974
        * alphat**-0.416
        * Z_eff**0.178
    )
    return c_bs * np.sqrt(eps) * beta_p


@relation(name="Bootstrap fraction Sugiyama L-mode", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_sugiyama_l_mode(
    eps: float, beta_p: float, alphan: float, alphat: float, Z_eff: float, q95: float, q0: float
) -> float:
    """Bootstrap current fraction, Sugiyama et al L-mode scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        0.740
        * eps**0.418
        * beta_p**0.904
        * alphan**0.06
        * alphat**-0.138
        * Z_eff**0.230
        * (q95 / q0) ** -0.142
    )


@relation(name="Bootstrap fraction Nevins", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_nevins(
    alphan: float, alphat: float, beta_T: float, B0: float, n_e_avg: float, I_p: float,
    q95: float, q0: float, R: float, a: float, T_e_avg: float, Z_eff: float, n0: float, T0: float,
) -> float:
    """Bootstrap current fraction, Nevins et al scaling (flux-surface integral).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS builds the local profile shape analytically from the density/
    temperature indices (parabolic) and integrates over the normalised minor
    radius with a definite quadrature. On-axis n0/T0 set the central electron
    beta; volume-averaged n/T set the local electron beta in the integrand.
    """
    # CHECK
    betae0 = n0 * T0 * 1.0e3 * ELECTRON_CHARGE_C / (B0**2 / (2.0 * MU0))

    def _integrand(y: float) -> float:
        betae = n_e_avg * T_e_avg * 1.0e3 * ELECTRON_CHARGE_C / (B0**2 / (2.0 * MU0))
        nabla = a * np.sqrt(y) / R
        x = (1.46 * np.sqrt(nabla) + 2.4 * nabla) / (1.0 - nabla) ** 1.5
        z = Z_eff
        d = (
            1.414 * z + z**2
            + x * (0.754 + 2.657 * z + (2.0 * z**2))
            + (x**2 * (0.348 + 1.243 * z + z**2))
        )
        a1 = (alphan + alphat) * (1.0 - y) ** (alphan + alphat - 1.0)
        a2 = alphat * (1.0 - y) ** (alphan + alphat - 1.0)
        al1 = (x / d) * (0.754 + 2.21 * z + z**2 + x * (0.348 + 1.243 * z + z**2))
        al2 = -x * ((0.884 + 2.074 * z) / d)
        alphai = -1.172 / (1.0 + 0.462 * x)
        q = q0 + (q95 - q0) * ((y + y**2 + y**3) / 3.0)
        pratio = (beta_T - betae) / betae
        return (q / q95) * (al1 * (a1 + (pratio * (a1 + alphai * a2))) + al2 * a2)

    ainteg, _ = integrate.quad(_integrand, 0.0, 1.0)
    aibs = 2.5 * betae0 * R * B0 * q95 * ainteg
    return 1.0e6 * aibs / I_p


@relation(name="Bootstrap fraction Wilson", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_wilson(
    alphaj: float, alphap: float, alphat: float, beta_thermal_poloidal: float,
    q0: float, q95: float, R: float, a: float,
) -> float:
    """Bootstrap current fraction, Wilson et al numerically-fitted scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS raises on illegal (NaN/negative) profile indices; that guard is
    dropped here -- a non-finite result propagates and is handled by the solver.
    """
    # CHECK
    term1 = np.log(0.5)
    term2 = np.log(q0 / q95)
    termp = 1.0 - 0.5 ** (1.0 / alphap)
    termt = 1.0 - 0.5 ** (1.0 / alphat)
    termj = 1.0 - 0.5 ** (1.0 / alphaj)

    alfpnw = term1 / np.log(np.log((q0 + (q95 - q0) * termp) / q95) / term2)
    alftnw = term1 / np.log(np.log((q0 + (q95 - q0) * termt) / q95) / term2)
    aj = term1 / np.log(np.log((q0 + (q95 - q0) * termj) / q95) / term2)

    z = 1.0
    r2 = R + a
    r1 = R - a
    eps1 = (r2 - r1) / (r2 + r1)
    saj = np.sqrt(aj)

    a_coeff = np.array([
        1.41 * (1.0 - 0.28 * saj) * (1.0 + 0.12 / z),
        0.36 * (1.0 - 0.59 * saj) * (1.0 + 0.8 / z),
        -0.27 * (1.0 - 0.47 * saj) * (1.0 + 3.0 / z),
        0.0053 * (1.0 + 5.0 / z),
        -0.93 * (1.0 - 0.34 * saj) * (1.0 + 0.15 / z),
        -0.26 * (1.0 - 0.57 * saj) * (1.0 - 0.27 * z),
        0.064 * (1.0 - 0.6 * aj + 0.15 * aj * aj) * (1.0 + 7.6 / z),
        -0.0011 * (1.0 + 9.0 / z),
        -0.33 * (1.0 - aj + 0.33 * aj * aj),
        -0.26 * (1.0 - 0.87 / saj - 0.16 * aj),
        -0.14 * (1.0 - 1.14 / saj - 0.45 * saj),
        -0.0069,
    ])
    seps1 = np.sqrt(eps1)
    b_coeff = np.array([
        1.0, alfpnw, alftnw, alfpnw * alftnw, seps1, alfpnw * seps1, alftnw * seps1,
        alfpnw * alftnw * seps1, eps1, alfpnw * eps1, alftnw * eps1, alfpnw * alftnw * eps1,
    ])
    return seps1 * beta_thermal_poloidal * (a_coeff * b_coeff).sum()


@relation(name="Bootstrap fraction Sugiyama H-mode", tags=_TAGS, outputs="f_BS")
def bootstrap_fraction_sugiyama_h_mode(
    eps: float, beta_p: float, alphan: float, alphat: float, tbeta: float, Z_eff: float,
    q95: float, q0: float, radius_plasma_pedestal_density_norm: float,
    nd_plasma_pedestal_electron: float, n_GW: float, temp_plasma_pedestal_kev: float,
) -> float:
    """Bootstrap current fraction, Sugiyama et al H-mode scaling.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Needs pedestal parameters (pedestal radius/density/temperature) that fusdb's
    parabolic-peaking profile model does not produce; supply them as inputs.
    """
    # CHECK
    return (
        0.789
        * eps**0.606
        * beta_p**0.960
        * alphan**0.0319
        * alphat**0.00822
        * tbeta**-0.0783
        * Z_eff**0.241
        * (q95 / q0) ** -0.103
        * radius_plasma_pedestal_density_norm**0.367
        * (nd_plasma_pedestal_electron / n_GW) ** -0.174
        * temp_plasma_pedestal_kev**0.0552
    )
