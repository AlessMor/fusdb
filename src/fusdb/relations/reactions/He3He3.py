"""He3He3 reactivity and reaction-rate relations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import constants as scipy_constants
from scipy.integrate import trapezoid

from fusdb import relation

from ...registry.tabulated_reactivities import reactivity_from_xsection_table


_AVOGADRO_NUMBER = scipy_constants.Avogadro
_KEV_TO_T9 = scipy_constants.kilo * scipy_constants.electron_volt / scipy_constants.Boltzmann / 1.0e9


@relation(
    name='He3He3 reactivity CF88',
    tags=('fusion_power',),
    outputs='sigmav_He3He3',
)
def sigmav_He3He3_CF88(T_i: float64 | NDArray[np.float64]) -> Any:
    """Return He3He3 reactivity from the CF88 parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The He3He3 reactivity in m^3/s.
    """
    # Convert the input temperature into the CF88 T9 variable.
    T9 = T_i * _KEV_TO_T9

    # Evaluate the CF88 parametrization in cm^3 mol^-1 s^-1.
    sigmav = (
        6.04e10
        / (T9 ** (2 / 3))
        * np.exp(-12.276 / (T9 ** (1 / 3)))
        * (
            1
            + 0.034 * (T9 ** (1 / 3))
            - 0.522 * (T9 ** (2 / 3))
            - 0.124 * T9
            + 0.353 * (T9 ** (4 / 3))
            + 0.213 * (T9 ** (5 / 3))
        )
    )

    # Convert from molar units to m^3/s.
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name='He3He3 reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_He3He3',
)
def sigmav_He3He3_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return He3He3 reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The He3He3 reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.0 He3He3 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("He3He3_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='He3He3 reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_He3He3',
)
def sigmav_He3He3_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return He3He3 reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The He3He3 reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.1 He3He3 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("He3He3_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='He3-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_He3He3',
)
def reaction_rate_he3he3(n_He3: float, sigmav_He3He3: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated He3He3 reaction rate.

    Args:
        n_He3: Helium-3 density profile.
        sigmav_He3He3: He3He3 reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total He3He3 reaction rate in 1/s.
    """
    # Form the local He3He3 reaction-rate density.
    integrand = 0.5 * (n_He3**2) * sigmav_He3He3

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
