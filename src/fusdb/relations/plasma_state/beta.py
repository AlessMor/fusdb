"""Plasma beta state relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J, MU0


@relation(
    name='Toroidal beta',
    tags=('plasma',),
    
    outputs='beta_T',
)
def toroidal_beta(p_th: float, B0: float) -> Any:
    """Freidberg Eq. 11.58: toroidal field contribution (dimensionless)."""
    return (2 * MU0 * p_th) / (B0 ** 2)


@relation(
    name='Poloidal beta',
    tags=('plasma',),
    
    outputs='beta_p',
)
def poloidal_beta(p_th: float, B_p: float) -> Any:
    """Freidberg Eq. 11.58 using poloidal field magnitude."""
    return (2 * MU0 * p_th) / (B_p ** 2)


@relation(
    name='Poloidal beta (cfspopcon)',
    tags=('plasma', 'tokamak'),
    outputs='beta_p',
)
def poloidal_beta_cfspopcon(p_th: float, B_pol_out_mid: float) -> Any:
    """Poloidal beta on cfspopcon's convention.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Same ``2 mu0 <p> / B_pol^2`` form as :func:`poloidal_beta`, but cfspopcon
    normalises to the *outboard-midplane* poloidal field
    ``B_pol_out_mid = mu0 I_p / (2 pi a)`` -- the circular-cross-section average,
    which ignores elongation.  fusdb's default ``B_p`` instead uses the
    elongation-corrected poloidal-field magnitude, which is smaller and therefore
    gives a ~2x larger ``beta_p``.  Selecting this relation reproduces cfspopcon's
    ``beta_poloidal`` (and with it the bootstrap fraction, which scales with it).

    Args:
        p_th: [Pa] volume-averaged thermal pressure
        B_pol_out_mid: [T] poloidal field at the outboard midplane

    Returns:
        beta_p [~]
    """
    return (2 * MU0 * p_th) / (B_pol_out_mid ** 2)


@relation(
    name='Beta decomposition',
    tags=('plasma',),
    
    outputs='beta',
)
def beta_decomposition(beta_T: float, beta_p: float) -> Any:
    """Total beta from toroidal and poloidal components (Freidberg Eq. 11.59)."""
    return 1 / (1 / beta_T + 1 / beta_p)


@relation(
    name='Normalized beta',
    tags=('plasma',),
    
    outputs='beta_N',
)
def normalized_beta(beta_T: float, a: float, B0: float, I_p: float) -> Any:
    """Wesson/Troyon normalized beta stored as a fraction.

    Literature often quotes this number in percent-style units, e.g. beta_N=3.9.
    FusDB stores the same quantity as 0.039 so it has the same fractional
    convention as beta_T.
    """
    I_p_MA = I_p / 1e6
    return beta_T * a * B0 / I_p_MA


@relation(
    name='Normalized beta (cfspopcon)',
    tags=('plasma', 'tokamak'),
    outputs='beta_N',
)
def normalized_beta_cfspopcon(beta: float, a: float, B0: float, I_p: float) -> Any:
    """Normalized beta on cfspopcon's convention (normalises the TOTAL beta).

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Same Troyon normalisation ``beta / (I_p / (a B0))`` as :func:`normalized_beta`,
    but cfspopcon normalises ``beta_total`` (the toroidal/poloidal harmonic
    combination) rather than ``beta_T``.  Since ``beta_total < beta_T``, fusdb's
    default runs higher; selecting this reproduces cfspopcon's ``normalized_beta``.
    """
    # CHECK
    I_p_MA = I_p / 1e6
    return beta * a * B0 / I_p_MA


@relation(name="Dipole inner beta", tags=("dipole", "plasma"), outputs="beta_in")
def dipole_inner_beta(n_e: Any, T_e: Any, n_i: Any, T_i: Any, B: Any) -> Any:
    """Levitated-dipole beta evaluated on the inner shell.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    p = (np.asarray(n_e)[..., 0] * np.asarray(T_e)[..., 0] + np.asarray(n_i)[..., 0] * np.asarray(T_i)[..., 0]) * KEV_TO_J
    return 2.0 * MU0 * p / np.asarray(B)[..., 0] ** 2


@relation(name="Dipole outer beta", tags=("dipole", "plasma"), outputs="beta_out")
def dipole_outer_beta(n_e: Any, T_e: Any, n_i: Any, T_i: Any, B: Any) -> Any:
    """Levitated-dipole beta evaluated on the outer shell.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    p = (np.asarray(n_e)[..., -1] * np.asarray(T_e)[..., -1] + np.asarray(n_i)[..., -1] * np.asarray(T_i)[..., -1]) * KEV_TO_J
    return 2.0 * MU0 * p / np.asarray(B)[..., -1] ** 2


@relation(name="Mirror peak beta (VSC)", tags=("mirror", "plasma"), outputs="beta")
def mirror_peak_beta_vsc(pres_plasma_on_axis: Any, B_vac: Any) -> Any:
    """Mirror peak beta referenced to the central-cell vacuum field.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return 2.0 * MU0 * np.asarray(pres_plasma_on_axis) / np.asarray(B_vac) ** 2
