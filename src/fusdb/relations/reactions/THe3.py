"""THe3 reactivity and reaction-rate relations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import constants as scipy_constants
from scipy.integrate import trapezoid

from fusdb import relation

from ...registry.tabulated_reactivities import reactivity_from_reactivity_table
from ...registry.tabulated_reactivities import reactivity_from_xsection_table


_AVOGADRO_NUMBER = scipy_constants.Avogadro
_KEV_TO_T9 = scipy_constants.kilo * scipy_constants.electron_volt / scipy_constants.Boltzmann / 1.0e9


@relation(
    name='THe3_D reactivity CF88',
    tags=('fusion_power',),
    outputs='sigmav_THe3_D',
)
def sigmav_THe3_D_CF88(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return THe3-to-D branch reactivity from the CF88 parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The THe3_D reactivity in m^3/s.
    """
    # Convert the input temperature into the CF88 variables.
    T9 = T_i * _KEV_TO_T9
    T9A = T9 / (1 + 0.128 * T9)

    # Evaluate the CF88 branch fit in cm^3 mol^-1 s^-1.
    sigmav = 5.46e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * np.exp(-7.733 / (T9A ** (1 / 3)))

    # Convert from molar units to m^3/s.
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name='THe3_D reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_THe3_D',
)
def sigmav_THe3_D_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return THe3-to-D branch reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The THe3_D reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.0 THe3_D cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("THe3D_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='THe3_D reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_THe3_D',
)
def sigmav_THe3_D_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return THe3-to-D branch reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The THe3_D reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.1 THe3_D cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("THe3D_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='THe3_D reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_THe3_D',
)
def sigmav_THe3_D_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return THe3-to-D branch reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The THe3_D reactivity in m^3/s.
    """
    # Interpolate the total THe3 NRL rate and apply the implemented branch fraction.
    return (
        reactivity_from_reactivity_table(
            "THe3_total_reactivity_NRL.yaml",
            T_i,
            interpolation_kind=interpolation_kind,
        )
        * 0.43
    )


@relation(
    name='THe3_np reactivity CF88',
    tags=('fusion_power',),
    outputs='sigmav_THe3_np',
)
def sigmav_THe3_np_CF88(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return THe3-to-np branch reactivity from the CF88 parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The THe3_np reactivity in m^3/s.
    """
    # Convert the input temperature into the CF88 variables.
    T9 = T_i * _KEV_TO_T9
    T9A = T9 / (1 + 0.115 * T9)

    # Evaluate the CF88 branch fit in cm^3 mol^-1 s^-1.
    sigmav = 7.71e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * np.exp(-7.733 / (T9A ** (1 / 3)))

    # Convert from molar units to m^3/s.
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name='THe3_np reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_THe3_np',
)
def sigmav_THe3_np_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return THe3-to-np branch reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The THe3_np reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.0 THe3_np cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("THe3n_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='THe3_np reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_THe3_np',
)
def sigmav_THe3_np_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return THe3-to-np branch reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The THe3_np reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.1 THe3_np cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("THe3n_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='THe3_np reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_THe3_np',
)
def sigmav_THe3_np_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return THe3-to-np branch reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The THe3_np reactivity in m^3/s.
    """
    # Interpolate the total THe3 NRL rate and apply the implemented branch fraction.
    return (
        reactivity_from_reactivity_table(
            "THe3_total_reactivity_NRL.yaml",
            T_i,
            interpolation_kind=interpolation_kind,
        )
        * 0.51
    )


@relation(
    name='THe3 reactivity CF88',
    tags=('fusion_power',),
    outputs='sigmav_THe3',
)
def sigmav_THe3_CF88(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total THe3 reactivity from the CF88 branch fits.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total THe3 reactivity in m^3/s.
    """
    # Sum the two implemented CF88 THe3 branches.
    return sigmav_THe3_np_CF88.func(T_i=T_i) + sigmav_THe3_D_CF88.func(T_i=T_i)


@relation(
    name='THe3 reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_THe3',
)
def sigmav_THe3_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total THe3 reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total THe3 reactivity in m^3/s.
    """
    # Sum the two implemented ENDF/B-VIII.0 THe3 branches.
    return sigmav_THe3_np_ENDFB_VIII0.func(T_i=T_i) + sigmav_THe3_D_ENDFB_VIII0.func(T_i=T_i)


@relation(
    name='THe3 reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_THe3',
)
def sigmav_THe3_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total THe3 reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total THe3 reactivity in m^3/s.
    """
    # Sum the two implemented ENDF/B-VIII.1 THe3 branches.
    return sigmav_THe3_np_ENDFB_VIII1.func(T_i=T_i) + sigmav_THe3_D_ENDFB_VIII1.func(T_i=T_i)


@relation(
    name='THe3 reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_THe3',
)
def sigmav_THe3_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return total THe3 reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The total THe3 reactivity in m^3/s.
    """
    # Interpolate the tabulated total THe3 reactivity data.
    return reactivity_from_reactivity_table(
        "THe3_total_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name='T-He3 alpha+D reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3_D',
)
def reaction_rate_the3_d(n_T: float, n_He3: float, sigmav_THe3_D: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated THe3-to-D reaction rate.

    Args:
        n_T: Tritium density profile.
        n_He3: Helium-3 density profile.
        sigmav_THe3_D: THe3_D reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total THe3_D reaction rate in 1/s.
    """
    # Form the local THe3_D reaction-rate density.
    integrand = n_T * n_He3 * sigmav_THe3_D

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)


@relation(
    name='T-He3 alpha+n+p reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3_np',
)
def reaction_rate_the3_np(n_T: float, n_He3: float, sigmav_THe3_np: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated THe3-to-np reaction rate.

    Args:
        n_T: Tritium density profile.
        n_He3: Helium-3 density profile.
        sigmav_THe3_np: THe3_np reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total THe3_np reaction rate in 1/s.
    """
    # Form the local THe3_np reaction-rate density.
    integrand = n_T * n_He3 * sigmav_THe3_np

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)


@relation(
    name='T-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_THe3',
)
def reaction_rate_the3(Rr_THe3_D: float, Rr_THe3_np: float) -> Any:
    """Return the total THe3 reaction rate from the implemented branches.

    Args:
        Rr_THe3_D: THe3-to-D branch reaction rate.
        Rr_THe3_np: THe3-to-np branch reaction rate.

    Returns:
        The total THe3 reaction rate in 1/s.
    """
    # Sum the implemented THe3 branch rates.
    return Rr_THe3_D + Rr_THe3_np
