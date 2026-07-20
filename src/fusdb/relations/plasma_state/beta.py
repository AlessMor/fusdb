"""Plasma beta state relations."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry import MU0


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
