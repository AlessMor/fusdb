"""Sauter-Angioni-Lin-Liu neoclassical bootstrap current model (PROCESS).

Ported from PROCESS ``process/models/physics/bootstrap_current.py``
(``SauterBootstrapCurrent``). The per-flux-surface model consumes fusdb's
electron/ion density and temperature profiles (``n_e``, ``T_e``, ``n_i``,
``T_i``) and their gradients, computes the L31/L32/L34 transport coefficients
with arbitrary-collisionality corrections, and integrates the bootstrap current
density over the (circularised) cross-section to give ``f_BS``. Gated so the
cfspopcon Gi scaling stays the default.

Reference: O. Sauter, C. Angioni, Y. R. Lin-Liu, Phys. Plasmas 6 (1999) 2834
(+ erratum, 9 (2002) 5140). Code supplied by E. Fable (IPP Garching).

Differences from PROCESS, all faithful to the same physics:
* PROCESS approximates the ion profiles as the electron profile scaled by the
  volume-averaged ion/electron ratio; fusdb passes its independent ``n_i``/
  ``T_i`` profiles directly.
* Only the default (ASTRA, ``fit=0``) trapped-particle-fraction is ported; the
  Sauter-2002/2016 variants (``fit=1``/``2``) are the enum branches and are not
  imported.
"""

import numpy as np

from fusdb.relation import relation


def _coulomb_logarithm(re, tempe, ne):
    return 15.9 - 0.5 * np.log(ne[re - 1]) + np.log(tempe[re - 1])


def _electron_collisions(re, tempe, ne):
    return (
        670.0
        * _coulomb_logarithm(re, tempe, ne)
        * ne[re - 1]
        / (tempe[re - 1] * np.sqrt(tempe[re - 1]))
    )


def _electron_collisionality(re, rmajor, zeff, inverse_q, sqeps, tempe, ne):
    return (
        _electron_collisions(re, tempe, ne)
        * 1.4
        * zeff[re - 1]
        * rmajor
        / np.abs(inverse_q[re - 1] * (sqeps[re - 1] ** 3) * np.sqrt(tempe[re - 1]) * 1.875e7)
    )


def _ion_collisions(re, zeff, ni, tempi, amain):
    return (
        zeff[re - 1] ** 4
        * ni[re - 1]
        * 322.0
        / (tempi[re - 1] * np.sqrt(tempi[re - 1] * amain[re - 1]))
    )


def _ion_collisionality(re, rmajor, inverse_q, sqeps, tempi, amain, zeff, ni):
    return (
        3.2e-6
        * _ion_collisions(re, zeff, ni, tempi, amain)
        * rmajor
        / (np.abs(inverse_q[re - 1] + 1.0e-4) * sqeps[re - 1] ** 3 * np.sqrt(tempi[re - 1] / amain[re - 1]))
    )


def _trapped_particle_fraction(re, triang, sqeps):
    # ASTRA method (PROCESS fit=0); h term excluded (dominates for eps < 0.5).
    sqeps_reduced = sqeps[re - 1]
    eps = sqeps_reduced**2
    zz = 1.0 - eps
    return 1.0 - zz * np.sqrt(zz) / (1.0 + 1.46 * sqeps_reduced)


def _beta_poloidal(re, nr, rmajor, b0, ne, tempe, inverse_q, rho):
    return (
        np.where(
            re != nr,
            1.6e-4 * np.pi * (ne[re] + ne[re - 1]) * (tempe[re] + tempe[re - 1]),
            6.4e-4 * np.pi * ne[re - 1] * tempe[re - 1],
        )
        * (rmajor / (b0 * rho[re - 1] * np.abs(inverse_q[re - 1] + 1.0e-4))) ** 2
    )


def _beta_poloidal_total(re, nr, rmajor, b0, ne, ni, tempe, tempi, inverse_q, rho):
    return (
        np.where(
            re != nr,
            1.6e-4
            * np.pi
            * (
                (ne[re] + ne[re - 1]) * (tempe[re] + tempe[re - 1])
                + (ni[re] + ni[re - 1]) * (tempi[re] + tempi[re - 1])
            ),
            6.4e-4 * np.pi * (ne[re - 1] * tempe[re - 1] + ni[re - 1] * tempi[re - 1]),
        )
        * (rmajor / (b0 * rho[re - 1] * np.abs(inverse_q[re - 1] + 1.0e-4))) ** 2
    )


def _l31_coefficient(re, nr, rmajor, b0, triang, ne, ni, tempe, tempi, inverse_q, rho, zeff, sqeps):
    z = zeff[re - 1]
    f_trapped = _trapped_particle_fraction(re, triang, sqeps)
    nu_e = _electron_collisionality(re, rmajor, zeff, inverse_q, sqeps, tempe, ne)
    f31_teff = f_trapped / (
        (1.0 + (1.0 - 0.1 * f_trapped) * np.sqrt(nu_e))
        + (0.5 * (1.0 - f_trapped) * nu_e) / z
    )
    l31 = (
        ((1.0 + 1.4 / (z + 1.0)) * f31_teff)
        - (1.9 / (z + 1.0) * f31_teff**2)
        + ((0.3 * f31_teff**3 + 0.2 * f31_teff**4) / (z + 1.0))
    )
    return l31 * _beta_poloidal_total(re, nr, rmajor, b0, ne, ni, tempe, tempi, inverse_q, rho)


def _l31_32_coefficient(re, nr, rmajor, b0, triang, ne, ni, tempe, tempi, inverse_q, rho, zeff, sqeps):
    z = zeff[re - 1]
    f_trapped = _trapped_particle_fraction(re, triang, sqeps)
    nu_e = _electron_collisionality(re, rmajor, zeff, inverse_q, sqeps, tempe, ne)
    f32ee_teff = f_trapped / (
        1.0
        + 0.26 * (1.0 - f_trapped) * np.sqrt(nu_e)
        + (0.18 * (1.0 - 0.37 * f_trapped) * nu_e / np.sqrt(z))
    )
    f32ei_teff = f_trapped / (
        (1.0 + (1.0 + 0.6 * f_trapped) * np.sqrt(nu_e))
        + (0.85 * (1.0 - 0.37 * f_trapped) * nu_e * (1.0 + z))
    )
    big_f32ee = (
        ((0.05 + 0.62 * z) / z / (1.0 + 0.44 * z) * (f32ee_teff - f32ee_teff**4))
        + ((f32ee_teff**2 - f32ee_teff**4 - 1.2 * (f32ee_teff**3 - f32ee_teff**4)) / (1.0 + 0.22 * z))
        + (1.2 / (1.0 + 0.5 * z) * f32ee_teff**4)
    )
    big_f32ei = (
        (-(0.56 + 1.93 * z) / z / (1.0 + 0.44 * z) * (f32ei_teff - f32ei_teff**4))
        + (4.95 / (1.0 + 2.48 * z) * (f32ei_teff**2 - f32ei_teff**4 - 0.55 * (f32ei_teff**3 - f32ei_teff**4)))
        - (1.2 / (1.0 + 0.5 * z) * f32ei_teff**4)
    )
    bp = _beta_poloidal(re, nr, rmajor, b0, ne, tempe, inverse_q, rho)
    bpt = _beta_poloidal_total(re, nr, rmajor, b0, ne, ni, tempe, tempi, inverse_q, rho)
    l31 = _l31_coefficient(re, nr, rmajor, b0, triang, ne, ni, tempe, tempi, inverse_q, rho, zeff, sqeps)
    return bp * (big_f32ee + big_f32ei) + l31 * bp / bpt


def _l34_alpha_31_coefficient(
    re, nr, rmajor, b0, triang, inverse_q, sqeps, tempi, tempe, amain, zmain, ni, ne, rho, zeff
):
    z = zeff[re - 1]
    f_trapped = _trapped_particle_fraction(re, triang, sqeps)
    nu_e = _electron_collisionality(re, rmajor, zeff, inverse_q, sqeps, tempe, ne)
    f34_teff = f_trapped / (
        (1.0 + (1.0 - 0.1 * f_trapped) * np.sqrt(nu_e))
        + 0.5 * (1.0 - 0.5 * f_trapped) * nu_e / z
    )
    l34 = (
        ((1.0 + (1.4 / (z + 1.0))) * f34_teff)
        - ((1.9 / (z + 1.0)) * f34_teff**2)
        + ((0.3 / (z + 1.0)) * f34_teff**3)
        + ((0.2 / (z + 1.0)) * f34_teff**4)
    )
    alpha_0 = (-1.17 * (1.0 - f_trapped)) / (1.0 - (0.22 * f_trapped) - 0.19 * f_trapped**2)
    nu_i = _ion_collisionality(re, rmajor, inverse_q, sqeps, tempi, amain, zmain, ni)
    alpha = (
        (alpha_0 + (0.25 * (1.0 - f_trapped**2)) * np.sqrt(nu_i)) / (1.0 + (0.5 * np.sqrt(nu_i)))
        + (0.315 * nu_i**2 * f_trapped**6)
    ) / (1.0 + (0.15 * nu_i**2 * f_trapped**6))
    bp = _beta_poloidal(re, nr, rmajor, b0, ne, tempe, inverse_q, rho)
    bpt = _beta_poloidal_total(re, nr, rmajor, b0, ne, ni, tempe, tempi, inverse_q, rho)
    l31 = _l31_coefficient(re, nr, rmajor, b0, triang, ne, ni, tempe, tempi, inverse_q, rho, zeff, sqeps)
    return (bpt - bp) * (l34 * alpha) + l31 * (1.0 - bp / bpt)


@relation(
    name="Bootstrap fraction Sauter",
    tags=("plasma", "current_drive", "tokamak", "process"),
    outputs="f_BS",
)
def bootstrap_fraction_sauter(
    n_e, T_e, n_i, T_i, rho_minor, S_phi, R, a, B0, delta, q0, q95, Z_eff, I_p, afuel, f_He3=0.0,
):
    """Bootstrap current fraction from the Sauter-Angioni-Lin-Liu neoclassical
    model, integrated over fusdb's density/temperature profiles.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    Args:
        n_e, T_e, n_i, T_i: electron/ion density [m^-3] and temperature [keV] profiles
        rho_minor: normalized physical minor-radius mapping r/a on the common profile grid
        S_phi: plasma poloidal cross-sectional area [m^2]
        R, a: major/minor radius [m]; B0: toroidal field [T]; delta: triangularity
        q0, q95: safety factors; Z_eff: effective charge; I_p: plasma current [A]
        afuel: main-ion mass [amu]; f_He3: He-3 fuel fraction (sets main-ion charge)
    """
    # CHECK
    roa = np.asarray(rho_minor, dtype=float)
    nr = roa.size
    rho_local = np.sqrt(S_phi / np.pi) * roa            # circularised minor radius
    sqeps = np.sqrt(roa * (a / R))
    ne = np.asarray(n_e, dtype=float) * 1e-19           # PROCESS works in 1e19 m^-3
    ni = np.asarray(n_i, dtype=float) * 1e-19
    tempe = np.asarray(T_e, dtype=float)                # keV
    tempi = np.asarray(T_i, dtype=float)
    zeff = np.full(nr, float(Z_eff))
    inverse_q = 1.0 / (q0 + (q95 - q0) * roa**2)
    amain = np.full(nr, float(afuel))
    zmain = np.full(nr, 1.0 + float(f_He3))

    radial_elements = np.arange(2, nr)
    drho = rho_local[radial_elements] - rho_local[radial_elements - 1]
    da = 2 * np.pi * rho_local[radial_elements - 1] * drho

    dlogte_drho = np.gradient(np.log(tempe), rho_local)[radial_elements - 1]
    dlogti_drho = np.gradient(np.log(tempi), rho_local)[radial_elements - 1]
    dlogne_drho = np.gradient(np.log(ne), rho_local)[radial_elements - 1]

    jboot = (
        0.5
        * (
            _l31_coefficient(radial_elements, nr, R, B0, delta, ne, ni, tempe, tempi, inverse_q, rho_local, zeff, sqeps)
            * dlogne_drho
            + _l31_32_coefficient(radial_elements, nr, R, B0, delta, ne, ni, tempe, tempi, inverse_q, rho_local, zeff, sqeps)
            * dlogte_drho
            + _l34_alpha_31_coefficient(radial_elements, nr, R, B0, delta, inverse_q, sqeps, tempi, tempe, amain, zmain, ni, ne, rho_local, zeff)
            * dlogti_drho
        )
        * 1.0e6
        * (-B0 / (0.2 * np.pi * R) * rho_local[radial_elements - 1] * inverse_q[radial_elements - 1])
    )

    return float(np.sum(da * jboot, axis=0) / I_p)
