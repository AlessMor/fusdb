"""DD reactivity and reaction-rate relations."""

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
    name='DDn reactivity BoschHale',
    tags=('fusion_power',),
    outputs='sigmav_DDn',
)
def sigmav_DDn_BoschHale(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDn reactivity from the Bosch-Hale parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDn reactivity in m^3/s.
    """
    # Define the DDn branch coefficients.
    coefficients = (((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0)

    # Compute the corrected fit variables for the Bosch-Hale form.
    theta = T_i / (
        1
        - (
            (coefficients[2] * T_i + coefficients[4] * T_i**2 + coefficients[6] * T_i**3)
            / (1 + coefficients[3] * T_i + coefficients[5] * T_i**2 + coefficients[7] * T_i**3)
        )
    )
    eta = coefficients[0] / (theta ** (1.0 / 3.0))

    # Evaluate the DDn branch reactivity and convert to m^3/s.
    sigmav = coefficients[1] * theta * np.sqrt(eta / (937814.0 * (T_i**3.0))) * np.exp(-3.0 * eta)
    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DDp reactivity BoschHale',
    tags=('fusion_power',),
    outputs='sigmav_DDp',
)
def sigmav_DDp_BoschHale(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDp reactivity from the Bosch-Hale parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDp reactivity in m^3/s.
    """
    # Define the DDp branch coefficients.
    coefficients = (((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0)

    # Compute the corrected fit variables for the Bosch-Hale form.
    theta = T_i / (
        1
        - (
            (coefficients[2] * T_i + coefficients[4] * T_i**2 + coefficients[6] * T_i**3)
            / (1 + coefficients[3] * T_i + coefficients[5] * T_i**2 + coefficients[7] * T_i**3)
        )
    )
    eta = coefficients[0] / (theta ** (1.0 / 3.0))

    # Evaluate the DDp branch reactivity and convert to m^3/s.
    sigmav = coefficients[1] * theta * np.sqrt(eta / (937814.0 * (T_i**3.0))) * np.exp(-3.0 * eta)
    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DD reactivity BoschHale',
    tags=('fusion_power',),
    outputs='sigmav_DD',
)
def sigmav_DD_BoschHale(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total DD reactivity from the Bosch-Hale parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total DD reactivity in m^3/s.
    """
    # Sum the two implemented DD branches.
    return sigmav_DDn_BoschHale.func(T_i=T_i) + sigmav_DDp_BoschHale.func(T_i=T_i)


@relation(
    name='DDn reactivity Hively',
    tags=('fusion_power',),
    outputs='sigmav_DDn',
)
def sigmav_DDn_Hively(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDn reactivity from the Hively parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDn reactivity in m^3/s.
    """
    # Define the DDn Hively coefficients.
    coefficients = (-15.993842, -35.017640, -1.3689787e-2, 2.7089621e-4, -2.9441547e-6, 1.2841202e-8)

    # Evaluate the fitted exponent and convert to m^3/s.
    exponent = (
        coefficients[0] / T_i**0.3725
        + coefficients[1]
        + coefficients[2] * T_i
        + coefficients[3] * T_i**2.0
        + coefficients[4] * T_i**3.0
        + coefficients[5] * T_i**4.0
    )
    return np.exp(exponent) * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DDp reactivity Hively',
    tags=('fusion_power',),
    outputs='sigmav_DDp',
)
def sigmav_DDp_Hively(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDp reactivity from the Hively parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDp reactivity in m^3/s.
    """
    # Define the DDp Hively coefficients.
    coefficients = (-15.511891, -35.318711, -1.2904737e-2, 2.6797766e-4, -2.9198685e-6, 1.2748415e-8)

    # Evaluate the fitted exponent and convert to m^3/s.
    exponent = (
        coefficients[0] / T_i**0.3735
        + coefficients[1]
        + coefficients[2] * T_i
        + coefficients[3] * T_i**2.0
        + coefficients[4] * T_i**3.0
        + coefficients[5] * T_i**4.0
    )
    return np.exp(exponent) * 1e-6  # type: ignore[no-any-return]


@relation(
    name='DD reactivity Hively',
    tags=('fusion_power',),
    outputs='sigmav_DD',
)
def sigmav_DD_Hively(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total DD reactivity from the Hively parametrization.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total DD reactivity in m^3/s.
    """
    # Sum the two Hively DD branches.
    return sigmav_DDn_Hively.func(T_i=T_i) + sigmav_DDp_Hively.func(T_i=T_i)


@relation(
    name='DD reactivity NRL',
    tags=('fusion_power',),
    outputs='sigmav_DD',
)
def sigmav_DD_NRL(
    T_i: float64 | NDArray[np.float64],
    *,
    interpolation_kind: str = "pchip",
) -> Any:
    """Return total DD reactivity from the NRL tabulated rates.

    Args:
        T_i: Ion temperature profile in keV.
        interpolation_kind: Interpolation scheme for the tabulated data.

    Returns:
        The total DD reactivity in m^3/s.
    """
    # Interpolate the tabulated total DD reactivity data.
    return reactivity_from_reactivity_table(
        "DD_total_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name='DDn reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_DDn',
)
def sigmav_DDn_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDn reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDn reactivity in m^3/s.
    """
    # Integrate the DDn ENDF/B-VIII.0 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DDn_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='DDp reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_DDp',
)
def sigmav_DDp_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDp reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDp reactivity in m^3/s.
    """
    # Integrate the DDp ENDF/B-VIII.0 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DDp_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name='DD reactivity ENDFB-VIII0',
    tags=('fusion_power',),
    outputs='sigmav_DD',
)
def sigmav_DD_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total DD reactivity from ENDF/B-VIII.0 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total DD reactivity in m^3/s.
    """
    # Sum the two ENDF/B-VIII.0 DD branches.
    return sigmav_DDn_ENDFB_VIII0.func(T_i=T_i) + sigmav_DDp_ENDFB_VIII0.func(T_i=T_i)


@relation(
    name='DDn reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_DDn',
)
def sigmav_DDn_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDn reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDn reactivity in m^3/s.
    """
    # Integrate the DDn ENDF/B-VIII.1 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DDn_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='DDp reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_DDp',
)
def sigmav_DDp_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return DDp reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The DDp reactivity in m^3/s.
    """
    # Integrate the DDp ENDF/B-VIII.1 cross-section table over a Maxwellian.
    return reactivity_from_xsection_table("DDp_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name='DD reactivity ENDFB-VIII1',
    tags=('fusion_power',),
    outputs='sigmav_DD',
)
def sigmav_DD_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64],
) -> Any:
    """Return total DD reactivity from ENDF/B-VIII.1 cross sections.

    Args:
        T_i: Ion temperature profile in keV.

    Returns:
        The total DD reactivity in m^3/s.
    """
    # Sum the two ENDF/B-VIII.1 DD branches.
    return sigmav_DDn_ENDFB_VIII1.func(T_i=T_i) + sigmav_DDp_ENDFB_VIII1.func(T_i=T_i)


@relation(
    name='DD (He3+n) reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DDn',
)
def reaction_rate_ddn(n_D: float, sigmav_DDn: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DDn reaction rate.

    Args:
        n_D: Deuterium density profile.
        sigmav_DDn: DDn reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DDn reaction rate in 1/s.
    """
    # Form the local DDn reaction-rate density.
    integrand = 0.5 * (n_D**2) * sigmav_DDn

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)


@relation(
    name='DD (T+p) reaction rate',
    tags=('fusion_power',),
    outputs='Rr_DDp',
)
def reaction_rate_ddp(n_D: float, sigmav_DDp: float, V_p: float, rho: float) -> Any:
    """Return the volume-integrated DDp reaction rate.

    Args:
        n_D: Deuterium density profile.
        sigmav_DDp: DDp reactivity profile.
        V_p: Plasma volume.

    Returns:
        The total DDp reaction rate in 1/s.
    """
    # Form the local DDp reaction-rate density.
    integrand = 0.5 * (n_D**2) * sigmav_DDp

    # Integrate the profile over the plasma volume.
    return V_p * trapezoid(integrand, x=rho)
