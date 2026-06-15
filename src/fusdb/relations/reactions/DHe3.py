"""DHe3 reactivity and reaction-rate relations."""

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
    name='DHe3 reactivity BoschHale',
    tags=('fusion_power',),
    outputs='sigmav_DHe3',
)
def sigmav_DHe3_BoschHale(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    r"""Return DHe3 reactivity from the Bosch-Hale parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DHe3 reactivity in m^3/s.
    """
    # Define the Bosch-Hale coefficients for the DHe3 reaction.
    coefficients = [
        ((68.7508**2) / 4.0) ** (1.0 / 3.0),
        5.51036e-10,
        6.41918e-03,
        -2.02896e-03,
        -1.91080e-05,
        1.35776e-04,
        0,
        0,
    ]
    reduced_mass_energy = 1124572.0

    # Compute the corrected fit variables.
    theta = T_i / (
        1
        - (
            (coefficients[2] * T_i + coefficients[4] * T_i**2 + coefficients[6] * T_i**3)
            / (1 + coefficients[3] * T_i + coefficients[5] * T_i**2 + coefficients[7] * T_i**3.0)
        )
    )
    eta = coefficients[0] / (theta ** (1.0 / 3.0))

    # Evaluate the fit and convert from cm^3/s to m^3/s.
    sigmav = coefficients[1] * theta * np.sqrt(eta / (reduced_mass_energy * (T_i**3.0))) * np.exp(-3.0 * eta)
    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DHe3 reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_DHe3',
)
def sigmav_DHe3_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return DHe3 reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The DHe3 reactivity in m^3/s.
    """
    # Interpolate the tabulated DHe3 reactivity data.
    return reactivity_from_reactivity_table(
        "DHe3_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name='DHe3 reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_DHe3',
)
def sigmav_DHe3_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DHe3 reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DHe3 reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.0 DHe3 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DHe3_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='DHe3 reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_DHe3',
)
def sigmav_DHe3_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DHe3 reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DHe3 reactivity in m^3/s.
    """
    # Integrate the ENDF/B-VIII.1 DHe3 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DHe3_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='D-He3 reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DHe3',
)
def reaction_rate_dhe3(n_D: float, n_He3: float, sigmav_DHe3: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DHe3 reaction rate.

    Args:
        n_D: Deuterium density profile.
        n_He3: Helium-3 density profile.
        sigmav_DHe3: DHe3 reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DHe3 reaction rate in 1/s.
    """
    # Form the local DHe3 reaction-rate density.
    integrand = n_D * n_He3 * sigmav_DHe3

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
