"""TT reactivity and reaction-rate relations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import constants as scipy_constants

from fusdb_pyomo import relation

from ...registry.tabulated_reactivities import reactivity_from_reactivity_table
from ...registry.tabulated_reactivities import reactivity_from_xsection_table


_AVOGADRO_NUMBER = scipy_constants.Avogadro
_KEV_TO_T9 = scipy_constants.kilo * scipy_constants.electron_volt / scipy_constants.Boltzmann / 1.0e9


@relation(
    name='TT reactivity CF88',
    tags=('fusion_power',),
    outputs='sigmav_TT',
)
def sigmav_TT_CF88(T_i: float64 | NDArray[np.float64]) -> Any:
    """Return TT reactivity from the CF88 parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The TT reactivity in m^3/s.
    """
    # Convert the input temperature into the CF88 T9 variable.
    T9 = T_i * _KEV_TO_T9

    # Evaluate the CF88 parametrization in cm^3 mol^-1 s^-1.
    sigmav = (
        1.67e9
        / (T9 ** (2 / 3))
        * np.exp(-4.872 / (T9 ** (1 / 3)))
        * (
            1
            + 0.086 * (T9 ** (1 / 3))
            - 0.455 * (T9 ** (2 / 3))
            - 0.272 * T9
            + 0.148 * (T9 ** (4 / 3))
            + 0.225 * (T9 ** (5 / 3))
        )
    )

    # Convert from molar units to m^3/s.
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name='TT reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_TT',
)
def sigmav_TT_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return TT reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The TT reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.0 TT cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("TT_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='TT reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_TT',
)
def sigmav_TT_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return TT reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The TT reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.1 TT cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("TT_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='TT reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_TT',
)
def sigmav_TT_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return TT reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The TT reactivity in m^3/s.
    """
    # Interpolate the tabulated TT reactivity data.
    return reactivity_from_reactivity_table(
        "TT_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name='T-T reaction rate',
    tags=('fusion_power',),
    outputs='Rr_TT',
)
def reaction_rate_tt(n_T: float, sigmav_TT: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated TT reaction rate.

    Args:
        n_T: Tritium density profile.
        sigmav_TT: TT reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total TT reaction rate in 1/s.
    """
    # Form the local TT reaction-rate density.
    integrand = 0.5 * (n_T**2) * sigmav_TT

    # Integrate the profile over the plasma volume.
    return V_p * np.trapezoid(integrand, x=rho)
