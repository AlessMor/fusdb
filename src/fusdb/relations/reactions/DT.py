"""DT reactivity and reaction-rate relations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from fusdb import relation

from ...registry.tabulated_reactivities import reactivity_from_reactivity_table
from ...registry.tabulated_reactivities import reactivity_from_xsection_table


@relation(
    name='DT reactivity BoschHale',
    tags=('fusion_power',),
    outputs='sigmav_DT',
)
def sigmav_DT_BoschHale(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    r"""Return DT reactivity from the Bosch-Hale parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DT reactivity in m^3/s.
    """
    # Define the Bosch-Hale coefficients for the DT reaction.
    coefficients = [0.0, 1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5]
    gamow_coefficient = 34.3827
    reduced_mass_energy = 1124656

    # Compute the temperature-corrected fit variables.
    theta = T_i / (
        1
        - (T_i * (coefficients[2] + T_i * (coefficients[4] + T_i * coefficients[6])))
        / (1 + T_i * (coefficients[3] + T_i * (coefficients[5] + T_i * coefficients[7])))
    )
    eta = (gamow_coefficient**2 / (4 * theta)) ** (1 / 3)

    # Evaluate the Bosch-Hale reactivity and convert from cm^3/s to m^3/s.
    sigmav = coefficients[1] * theta * np.sqrt(eta / (reduced_mass_energy * T_i**3)) * np.exp(-3 * eta)
    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DT reactivity Hively',
    tags=('fusion_power',),
    outputs='sigmav_DT',
)
def sigmav_DT_Hively(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    r"""Return DT reactivity from the Hively parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DT reactivity in m^3/s.
    """
    # Define the Hively polynomial coefficients.
    coefficients = [-21.377692, -25.204054, -7.1013427e-2, 1.9375451e-4, 4.9246592e-6, -3.9836572e-8]
    exponent_power = 0.2935

    # Build the fitted exponent and convert the result to m^3/s.
    exponent = (
        coefficients[0] / T_i**exponent_power
        + coefficients[1]
        + coefficients[2] * T_i
        + coefficients[3] * T_i**2.0
        + coefficients[4] * T_i**3.0
        + coefficients[5] * T_i**4.0
    )
    return np.exp(exponent) * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DT reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_DT',
)
def sigmav_DT_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return DT reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The DT reactivity in m^3/s.
    """
    # Interpolate the tabulated DT reactivity data.
    return reactivity_from_reactivity_table(
        "DT_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name='DT reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_DT',
)
def sigmav_DT_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DT reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DT reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.0 DT cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DT_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='DT reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_DT',
)
def sigmav_DT_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DT reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DT reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.1 DT cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DT_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='DT reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DT',
)
def reaction_rate_dt(n_D: float, n_T: float, sigmav_DT: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DT reaction rate.

    Args:
        n_D: Deuterium density profile.
        n_T: Tritium density profile.
        sigmav_DT: DT reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DT reaction rate in 1/s.
    """
    # Form the local DT reaction-rate density.
    integrand = n_D * n_T * sigmav_DT

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
