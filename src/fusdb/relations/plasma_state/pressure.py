"""Plasma pressure state relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import line_average, volume_average
from fusdb.registry import KEV_TO_J


def _thermal_pressure_profile(n_e: Any, T_e: Any, n_i: Any, T_i: Any) -> Any:
    """Return the local thermal pressure profile in Pa."""
    return KEV_TO_J * (
        np.asarray(n_e, dtype=float) * np.asarray(T_e, dtype=float)
        + np.asarray(n_i, dtype=float) * np.asarray(T_i, dtype=float)
    )


@relation(
    name='Thermal pressure',
    tags=('plasma',),
    outputs='p_th',
)
def thermal_pressure(
    n_e: float,
    T_e: float,
    n_i: float,
    T_i: float,
    rho: float,
    w_V: Any = None,
) -> Any:
    """Return volume-averaged thermal pressure from profile/local quantities.

    ``rho`` is the computational sampling grid. ``w_V`` may supply the physical
    volume measure; omitting it retains the historical self-similar weighting.
    """
    return volume_average(
        _thermal_pressure_profile(n_e, T_e, n_i, T_i), rho, weight=w_V
    )


@relation(
    name='Average total pressure (cfspopcon)',
    tags=('plasma', 'tokamak'),
    outputs='p_th',
)
def average_total_pressure_cfspopcon(n_e_avg: float, T_e_avg: float, T_i_avg: float) -> Any:
    """Average total thermal pressure on cfspopcon's diagnostic convention.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    cfspopcon's ``average_total_pressure`` is the undiluted product of averages
    ``n_e <(T_e + T_i)>`` -- it uses the electron density for the ion term (no
    dilution) and does not integrate the peaked profiles.  fusdb's default
    :func:`thermal_pressure` instead flux-volume-integrates ``n_e T_e + n_i T_i``
    with the real diluted ``n_i``; the two differ ~13% here.  This diagnostic is
    inconsistent with cfspopcon's own stored energy (which does carry the
    dilution, and which fusdb matches via "Plasma stored energy (cfspopcon)"),
    so it is provided only to reproduce cfspopcon's beta chain, not as improved
    physics.
    """
    # CHECK
    return KEV_TO_J * n_e_avg * (T_e_avg + T_i_avg)


@relation(
    name='Thermal pressure rho-average',
    tags=('plasma',),
    outputs='p_th_rho_avg',
)
def thermal_pressure_rho_average(n_e: float, T_e: float, n_i: float, T_i: float, rho: float) -> Any:
    """Return straight-rho averaged thermal pressure from profile/local quantities."""
    return line_average(_thermal_pressure_profile(n_e, T_e, n_i, T_i), rho)


@relation(
    name='Peak pressure',
    tags=('plasma',),
    outputs='p_peak',
)
def peak_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> Any:
    """Calculate the peak pressure."""
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J
