"""Barr (2018) surface-inductance sub-model: external inductance and vertical field.

Adapted from cfspopcon (plasma_current/flux_consumption); see README.md section
"Third-party Notices". cfspopcon exposes Barr/Hirshman coefficient sets and four
vertical-field equations via enums; this port hardcodes the **Barr** coefficients
and the **Barr** vertical-field equation (the primary cite). The fa..fh helpers
are cfspopcon's, reduced from xarray ``apply_ufunc`` to scalar numpy.
"""

from typing import Any

import numpy as np

from fusdb import relation
from fusdb.registry import MU0

# Barr (2018) surface-inductance coefficients.
_A = np.array([1.438, 2.139, 9.387, -1.939])
_B = np.array([0.149, 1.068, -6.216, 4.126])
_C = np.array([-0.293, -0.349, 0.098])
_D = np.array([0.003, 0.334, -2.018])
_E = np.array([0.080, -0.260, -0.267, 1.135])


def _fa_sums(eps: Any) -> tuple[Any, Any]:
    s = np.sqrt(eps)
    sum1 = _A[0] * s + _A[1] * s**2
    sum2 = _A[2] * s + _A[3] * s**2
    return sum1, sum2


def _fa_sum_ne(eps: Any) -> Any:
    s = np.sqrt(eps)
    return _E[0] * s + _E[1] * s**2 + _E[2] * s**3 + _E[3] * s**4


def _fb_sum_nb(eps: Any) -> Any:
    return _B[1] * eps**4 + _B[2] * eps**5 + _B[3] * eps**6


def _fc_sum_nc(eps: Any) -> Any:
    return _C[0] * eps**2 + _C[1] * eps**4 + _C[2] * eps**6


def _fd_sum_nd(eps: Any) -> Any:
    return _D[1] * eps**1 + _D[2] * eps**2


def _fg_sums(eps: Any) -> tuple[Any, Any]:
    s = np.sqrt(eps)
    sum1 = 0.5 * _A[0] / s + _A[1]
    sum2 = (_A[0] + 0.5 * _A[2]) / s + (_A[1] + _A[3])
    return sum1, sum2


def _fg_sum_ce(eps: Any) -> Any:
    s = np.sqrt(eps)
    return 0.5 * _E[0] + _E[1] * s + 1.5 * _E[2] * s**2 + 2.0 * _E[3] * s**3


def _fh_sum_cb(eps: Any) -> Any:
    return 4.5 * _B[1] * eps**4 + 5.5 * _B[2] * eps**5 + 6.5 * _B[3] * eps**6


def _fa(eps: Any, beta_p: Any, li: Any) -> Any:
    sum1, sum2 = _fa_sums(eps)
    return ((1 + sum1) * np.log(8 / eps)) - (2 + sum2) + (beta_p + li / 2) * _fa_sum_ne(eps)


def _fb(eps: Any) -> Any:
    return _B[0] * np.sqrt(eps) * (1 + _fb_sum_nb(eps))


def _fc(eps: Any) -> Any:
    return 1 + _fc_sum_nc(eps)


def _fd(eps: Any) -> Any:
    return _D[0] * eps * (1 + _fd_sum_nd(eps))


def _fg(eps: Any, beta_p: Any, li: Any) -> Any:
    sum1, sum2 = _fg_sums(eps)
    return -(1 / eps) + np.log(8 / eps) * sum1 - sum2 + (beta_p + li / 2) * _fg_sum_ce(eps)


def _fh(eps: Any, kappa: Any) -> Any:
    return -1 + ((kappa * _B[0]) / np.sqrt(eps)) * (0.5 + _fh_sum_cb(eps))


@relation(
    name="External inductance (Barr)",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="external_inductance",
)
def calc_external_inductance(eps: Any, kappa: Any, beta_p: Any, R: Any, internal_inductivity: Any) -> Any:
    """External self-inductance of the plasma (Barr 2018, eq. 13).

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
        eps: [~] :term:`glossary link<inverse_aspect_ratio>`
        kappa: [~] :term:`glossary link<areal_elongation>`
        beta_p: [~] :term:`glossary link<beta_poloidal>`
        R: [m] :term:`glossary link<major_radius>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`

    Returns:
        external_inductance [H]
    """
    # CHECK
    fa = _fa(eps, beta_p, internal_inductivity)
    fb = _fb(eps)
    return MU0 * R * fa * (1 - eps) / ((1 - eps) + kappa * fb)


@relation(
    name="Vertical field mutual inductance (Barr)",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="vertical_field_mutual_inductance",
)
def calc_vertical_field_mutual_inductance(eps: Any, kappa: Any) -> Any:
    """Mutual inductance linking the surface to the vertical field (Barr 2018, eq. 15).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    fc = _fc(eps)
    fd = _fd(eps)
    return (1 - eps) ** 2 / ((1 - eps) ** 2 * fc + fd * np.sqrt(kappa))


@relation(
    name="Inverse-mu0 dLe/dR (Barr)",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="invmu_0_dLedR",
)
def calc_invmu_0_dLedR(
    eps: Any, kappa: Any, beta_p: Any, internal_inductivity: Any, external_inductance: Any, R: Any
) -> Any:
    """(1/mu_0) d(external inductance)/dR (Barr 2018, eq. 21).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    fa = _fa(eps, beta_p, internal_inductivity)
    fb = _fb(eps)
    fg = _fg(eps, beta_p, internal_inductivity)
    fh = _fh(eps, kappa)
    denom = (1 - eps) + kappa * fb
    return (1 / MU0) * (
        MU0 * eps * (1 - eps) * fa * fh / (denom**2)
        - MU0 * eps * (1 - eps) * fg / denom
        + eps * MU0 * fa / denom
        + external_inductance / R
    )


@relation(
    name="Vertical magnetic field (Barr)",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="vertical_magnetic_field",
)
def calc_vertical_magnetic_field(
    beta_p: Any, internal_inductivity: Any, R: Any, I_p: Any, invmu_0_dLedR: Any
) -> Any:
    """Vertical magnetic field for radial force balance (Barr 2018, eq. 16).

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    cfspopcon offers four vertical-field equations behind an enum; the Barr form
    is used here.
    """
    # CHECK
    return MU0 * I_p * (1 / (4 * np.pi * R)) * (invmu_0_dLedR + (beta_p + internal_inductivity / 2) - 0.5)
