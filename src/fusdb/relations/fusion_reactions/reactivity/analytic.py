"""Analytic fusion reactivity relations."""

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy import constants as scipy_constants

from fusdb.relation import relation


_AVOGADRO_NUMBER = scipy_constants.Avogadro
_KEV_TO_T9 = scipy_constants.kilo * scipy_constants.electron_volt / scipy_constants.Boltzmann / 1.0e9

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

    # Compute the temperature-corrected fit variables.  A parabolic profile
    # reaches T_i=0 at the edge, where theta=0 makes eta and 1/T_i**3 diverge
    # into a 0*inf NaN; the reactivity there is physically zero (no fusion at
    # zero temperature), so non-finite contributions are zeroed -- matching the
    # edge guard the synchrotron-radiation relation applies for the same reason.
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = T_i / (
            1
            - (T_i * (coefficients[2] + T_i * (coefficients[4] + T_i * coefficients[6])))
            / (1 + T_i * (coefficients[3] + T_i * (coefficients[5] + T_i * coefficients[7])))
        )
        eta = (gamow_coefficient**2 / (4 * theta)) ** (1 / 3)

        # Evaluate the Bosch-Hale reactivity and convert from cm^3/s to m^3/s.
        sigmav = coefficients[1] * theta * np.sqrt(eta / (reduced_mass_energy * T_i**3)) * np.exp(-3 * eta)
    sigmav = np.where(np.asarray(T_i, dtype=float) > 0.0, sigmav, 0.0)
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
