import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray
from scipy import constants as scipy_constants

from fusdb.relation_util import relation
from fusdb.relations.reactivities.tabulated_reactivities import reactivity_from_reactivity_table
from fusdb.relations.reactivities.tabulated_reactivities import reactivity_from_xsection_table


_AVOGADRO_NUMBER = scipy_constants.Avogadro
_KEV_TO_T9 = scipy_constants.kilo * scipy_constants.electron_volt / scipy_constants.Boltzmann / 1.0e9


# TODO(high): check ranges of validity for each function


# %%%%%%%%%%%%%%%% DT %%%%%%%%%%%%%%%%

@relation(
    name="DT reactivity BoschHale",
    output="sigmav_DT",
    tags=("fusion_power",),
)
def sigmav_DT_BoschHale(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    :func:`sigmav_DT_BoschHale` is more accurate than :func:`sigmav_DT` for ion_temp_profile > ~48.45 keV (estimate based on
    linear interp between errors found at available datapoints).
    Maximum error = 1.4% within range 50-1000 keV from available NRL data.

    Formulation from :cite:`bosch_improved_1992`

    Args:
        T_i: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s.

    """

    # Bosch Hale coefficients for DT reaction
    C = [0.0, 1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5]
    B_G = 34.3827
    mr_c2 = 1124656

    theta = T_i / (
        1
        - (T_i * (C[2] + T_i * (C[4] + T_i * C[6])))
        / (1 + T_i * (C[3] + T_i * (C[5] + T_i * C[7])))
    )
    eta = (B_G**2 / (4 * theta)) ** (1 / 3)
    math = sp if isinstance(T_i, sp.Expr) else np
    sigmav = C[1] * theta * math.sqrt(eta / (mr_c2 * T_i**3)) * math.exp(-3 * eta)

    return sigmav * 1e-6  # type: ignore[no-any-return] # [m^3/s]


@relation(
    name="DT reactivity Hively",
    output="sigmav_DT",
    tags=("fusion_power",),
)
def sigmav_DT_Hively(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    r"""DT reactivity from the Hively 1977 parametrization.

    The original fit returns :math:`\langle \sigma v \rangle` in cm^3/s.
    This implementation converts the result to m^3/s.
    """
    coefficients = [-21.377692, -25.204054, -7.1013427e-2, 1.9375451e-4, 4.9246592e-6, -3.9836572e-8]
    exponent_power = 0.2935
    math = sp if isinstance(T_i, sp.Expr) else np
    exponent = (
        coefficients[0] / T_i**exponent_power
        + coefficients[1]
        + coefficients[2] * T_i
        + coefficients[3] * T_i**2.0
        + coefficients[4] * T_i**3.0
        + coefficients[5] * T_i**4.0
    )
    return math.exp(exponent) * 1e-6  # type: ignore[no-any-return]


@relation(
    name="DT reactivity NRL",
    output="sigmav_DT",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_DT_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DT reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "DT_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name="DT reactivity ENDFB-VIII0",
    output="sigmav_DT",
    tags=("fusion_power",),
)
def sigmav_DT_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DT reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("DT_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="DT reactivity ENDFB-VIII1",
    output="sigmav_DT",
    tags=("fusion_power",),
)
def sigmav_DT_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DT reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("DT_xsection_ENDFB-VIII1.yaml", T_i)


# %%%%%%%%%%%%%%%% DD %%%%%%%%%%%%%%%%


@relation(
    name="DDn reactivity BoschHale",
    output="sigmav_DDn",
    tags=("fusion_power",),
)
def sigmav_DDn_BoschHale(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (He3+n) branch reactivity from the Bosch-Hale parametrization."""
    coefficients = (((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0)
    theta = T_i / (
        1
        - (
            (coefficients[2] * T_i + coefficients[4] * T_i**2 + coefficients[6] * T_i**3)
            / (1 + coefficients[3] * T_i + coefficients[5] * T_i**2 + coefficients[7] * T_i**3)
        )
    )
    eta = coefficients[0] / (theta ** (1.0 / 3.0))
    math = sp if isinstance(T_i, sp.Expr) else np
    sigmav = coefficients[1] * theta * math.sqrt(eta / (937814.0 * (T_i**3.0))) * math.exp(-3.0 * eta)
    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name="DDp reactivity BoschHale",
    output="sigmav_DDp",
    tags=("fusion_power",),
)
def sigmav_DDp_BoschHale(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (T+p) branch reactivity from the Bosch-Hale parametrization."""
    coefficients = (((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0)
    theta = T_i / (
        1
        - (
            (coefficients[2] * T_i + coefficients[4] * T_i**2 + coefficients[6] * T_i**3)
            / (1 + coefficients[3] * T_i + coefficients[5] * T_i**2 + coefficients[7] * T_i**3)
        )
    )
    eta = coefficients[0] / (theta ** (1.0 / 3.0))
    math = sp if isinstance(T_i, sp.Expr) else np
    sigmav = coefficients[1] * theta * math.sqrt(eta / (937814.0 * (T_i**3.0))) * math.exp(-3.0 * eta)
    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name="DD reactivity BoschHale",
    output="sigmav_DD",
    tags=("fusion_power",),
)
def sigmav_DD_BoschHale(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total DD reactivity from the Bosch-Hale parametrization."""
    return sigmav_DDn_BoschHale(T_i) + sigmav_DDp_BoschHale(T_i)


@relation(
    name="DDn reactivity Hively",
    output="sigmav_DDn",
    tags=("fusion_power",),
)
def sigmav_DDn_Hively(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (He3+n) branch reactivity from the Hively parametrization."""
    coefficients = (-15.993842, -35.017640, -1.3689787e-2, 2.7089621e-4, -2.9441547e-6, 1.2841202e-8)
    math = sp if isinstance(T_i, sp.Expr) else np
    exponent = (
        coefficients[0] / T_i**0.3725
        + coefficients[1]
        + coefficients[2] * T_i
        + coefficients[3] * T_i**2.0
        + coefficients[4] * T_i**3.0
        + coefficients[5] * T_i**4.0
    )
    return math.exp(exponent) * 1e-6  # type: ignore[no-any-return]


@relation(
    name="DDp reactivity Hively",
    output="sigmav_DDp",
    tags=("fusion_power",),
)
def sigmav_DDp_Hively(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (T+p) branch reactivity from the Hively parametrization."""
    coefficients = (-15.511891, -35.318711, -1.2904737e-2, 2.6797766e-4, -2.9198685e-6, 1.2748415e-8)
    math = sp if isinstance(T_i, sp.Expr) else np
    exponent = (
        coefficients[0] / T_i**0.3735
        + coefficients[1]
        + coefficients[2] * T_i
        + coefficients[3] * T_i**2.0
        + coefficients[4] * T_i**3.0
        + coefficients[5] * T_i**4.0
    )
    return math.exp(exponent) * 1e-6  # type: ignore[no-any-return]


@relation(
    name="DD reactivity Hively",
    output="sigmav_DD",
    tags=("fusion_power",),
)
def sigmav_DD_Hively(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total DD reactivity from the Hively parametrization."""
    return sigmav_DDn_Hively(T_i) + sigmav_DDp_Hively(T_i)


@relation(
    name="DD reactivity NRL",
    output="sigmav_DD",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_DD_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total DD reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "DD_total_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name="DDn reactivity ENDFB-VIII0",
    output="sigmav_DDn",
    tags=("fusion_power",),
)
def sigmav_DDn_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (He3+n) branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("DDn_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="DDp reactivity ENDFB-VIII0",
    output="sigmav_DDp",
    tags=("fusion_power",),
)
def sigmav_DDp_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (T+p) branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("DDp_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="DD reactivity ENDFB-VIII0",
    output="sigmav_DD",
    tags=("fusion_power",),
)
def sigmav_DD_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total DD reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return sigmav_DDn_ENDFB_VIII0(T_i) + sigmav_DDp_ENDFB_VIII0(T_i)


@relation(
    name="DDn reactivity ENDFB-VIII1",
    output="sigmav_DDn",
    tags=("fusion_power",),
)
def sigmav_DDn_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (He3+n) branch reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("DDn_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name="DDp reactivity ENDFB-VIII1",
    output="sigmav_DDp",
    tags=("fusion_power",),
)
def sigmav_DDp_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (T+p) branch reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("DDp_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name="DD reactivity ENDFB-VIII1",
    output="sigmav_DD",
    tags=("fusion_power",),
)
def sigmav_DD_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total DD reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return sigmav_DDn_ENDFB_VIII1(T_i) + sigmav_DDp_ENDFB_VIII1(T_i)


# %%%%%%%%%%%%%%%% DHe3 %%%%%%%%%%%%%%%%

@relation(
    name="DHe3 reactivity BoschHale",
    output="sigmav_DHe3",
    tags=("fusion_power",),
)
def sigmav_DHe3_BoschHale(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    r"""Deuterium-Helium-3 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 8.4% within range 2-100 keV and should not be used outside range [2, 100] keV.

    Uses DD cross section formulation :cite:`bosch_improved_1992`.

    Args:
        T_i: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s.
    """
    # For He3(d,p)4He
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
    mc2 = 1124572.0

    theta = T_i / (
        1
        - (
            (coefficients[2] * T_i + coefficients[4] * T_i**2 + coefficients[6] * T_i**3)
            / (1 + coefficients[3] * T_i + coefficients[5] * T_i**2 + coefficients[7] * T_i**3.0)
        )
    )
    eta = coefficients[0] / (theta ** (1.0 / 3.0))

    math = sp if isinstance(T_i, sp.Expr) else np
    sigmav = coefficients[1] * theta * math.sqrt(eta / (mc2 * (T_i**3.0))) * math.exp(-3.0 * eta)

    return sigmav * 1e-6  # type: ignore[no-any-return]


@relation(
    name="DHe3 reactivity NRL",
    output="sigmav_DHe3",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_DHe3_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """D-He3 reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "DHe3_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


@relation(
    name="DHe3 reactivity ENDFB-VIII0",
    output="sigmav_DHe3",
    tags=("fusion_power",),
)
def sigmav_DHe3_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """D-He3 reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("DHe3_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="DHe3 reactivity ENDFB-VIII1",
    output="sigmav_DHe3",
    tags=("fusion_power",),
)
def sigmav_DHe3_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """D-He3 reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("DHe3_xsection_ENDFB-VIII1.yaml", T_i)


# %%%%%%%%%%%%%%%% TT %%%%%%%%%%%%%%%%

@relation(
    name="TT reactivity CF88",
    output="sigmav_TT",
    tags=("fusion_power",),
)
def sigmav_TT_CF88(T_i: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""Tritium-Tritium reaction using the CF88 parametrization."""
    T9 = T_i * _KEV_TO_T9
    math = sp if isinstance(T9, sp.Expr) else np
    sigmav = (
        1.67e9
        / (T9 ** (2 / 3))
        * math.exp(-4.872 / (T9 ** (1 / 3)))
        * (
            1
            + 0.086 * (T9 ** (1 / 3))
            - 0.455 * (T9 ** (2 / 3))
            - 0.272 * T9
            + 0.148 * (T9 ** (4 / 3))
            + 0.225 * (T9 ** (5 / 3))
        )
    )
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name="TT reactivity ENDFB-VIII0",
    output="sigmav_TT",
    tags=("fusion_power",),
)
def sigmav_TT_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """TT reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("TT_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="TT reactivity ENDFB-VIII1",
    output="sigmav_TT",
    tags=("fusion_power",),
)
def sigmav_TT_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """TT reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("TT_xsection_ENDFB-VIII1.yaml", T_i)


@relation(
    name="TT reactivity NRL",
    output="sigmav_TT",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_TT_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """TT reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "TT_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )


# %%%%%%%%%%%%%%%% He3He3 %%%%%%%%%%%%%%%%

@relation(
    name="He3He3 reactivity CF88",
    output="sigmav_He3He3",
    tags=("fusion_power",),
)
def sigmav_He3He3_CF88(T_i: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""Helium-3/Helium-3 reaction using the CF88 parametrization."""
    T9 = T_i * _KEV_TO_T9
    math = sp if isinstance(T9, sp.Expr) else np
    sigmav = (
        6.04e10
        / (T9 ** (2 / 3))
        * math.exp(-12.276 / (T9 ** (1 / 3)))
        * (
            1
            + 0.034 * (T9 ** (1 / 3))
            - 0.522 * (T9 ** (2 / 3))
            - 0.124 * T9
            + 0.353 * (T9 ** (4 / 3))
            + 0.213 * (T9 ** (5 / 3))
        )
    )
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name="He3He3 reactivity ENDFB-VIII0",
    output="sigmav_He3He3",
    tags=("fusion_power",),
)
def sigmav_He3He3_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """He3-He3 reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("He3He3_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="He3He3 reactivity ENDFB-VIII1",
    output="sigmav_He3He3",
    tags=("fusion_power",),
)
def sigmav_He3He3_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """He3-He3 reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("He3He3_xsection_ENDFB-VIII1.yaml", T_i)


# %%%%%%%%%%%%%%%% THe3 %%%%%%%%%%%%%%%%

@relation(
    name="THe3_D reactivity CF88",
    output="sigmav_THe3_D",
    tags=("fusion_power",),
)
def sigmav_THe3_D_CF88(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    r"""T + 3He -> 4He + D branch using the CF88 parametrization."""
    T9 = T_i * _KEV_TO_T9
    T9A = T9 / (1 + 0.128 * T9)
    math = sp if isinstance(T9, sp.Expr) else np
    # The CF88 shorthand is T9A^(5/6) * exp(-7.733 / T9A^(1/3)).
    sigmav = 5.46e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * math.exp(-7.733 / (T9A ** (1 / 3)))
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name="THe3_D reactivity ENDFB-VIII0",
    output="sigmav_THe3_D",
    tags=("fusion_power",),
)
def sigmav_THe3_D_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + D branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("THe3D_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="THe3_D reactivity ENDFB-VIII1",
    output="sigmav_THe3_D",
    tags=("fusion_power",),
)
def sigmav_THe3_D_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + D branch reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("THe3D_xsection_ENDFB-VIII1.yaml", T_i)

@relation(
    name="THe3_D reactivity NRL",
    output="sigmav_THe3_D",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_THe3_D_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + D branch reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "THe3_total_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )*0.43


@relation(
    name="THe3_np reactivity CF88",
    output="sigmav_THe3_np",
    tags=("fusion_power",),
)
def sigmav_THe3_np_CF88(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    r"""T + 3He -> 4He + n + p branch using the CF88 parametrization."""
    T9 = T_i * _KEV_TO_T9
    T9A = T9 / (1 + 0.115 * T9)
    math = sp if isinstance(T9, sp.Expr) else np
    # The CF88 shorthand is T9A^(5/6) * exp(-7.733 / T9A^(1/3)).
    sigmav = 7.71e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * math.exp(-7.733 / (T9A ** (1 / 3)))
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


@relation(
    name="THe3_np reactivity ENDFB-VIII0",
    output="sigmav_THe3_np",
    tags=("fusion_power",),
)
def sigmav_THe3_np_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + n + p branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return reactivity_from_xsection_table("THe3n_xsection_ENDFB-VIII0.yaml", T_i)


@relation(
    name="THe3_np reactivity ENDFB-VIII1",
    output="sigmav_THe3_np",
    tags=("fusion_power",),
)
def sigmav_THe3_np_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + n + p branch reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return reactivity_from_xsection_table("THe3n_xsection_ENDFB-VIII1.yaml", T_i)

@relation(
    name="THe3_np reactivity NRL",
    output="sigmav_THe3_np",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_THe3_np_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + n + p branch reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "THe3_total_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    ) * 0.51

@relation(
    name="THe3 reactivity CF88",
    output="sigmav_THe3",
    tags=("fusion_power",),
)
def sigmav_THe3_CF88(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total T-He3 reactivity from the CF88 branch parametrizations."""
    return sigmav_THe3_np_CF88(T_i) + sigmav_THe3_D_CF88(T_i)


@relation(
    name="THe3 reactivity ENDFB-VIII0",
    output="sigmav_THe3",
    tags=("fusion_power",),
)
def sigmav_THe3_ENDFB_VIII0(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total T-He3 reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return sigmav_THe3_np_ENDFB_VIII0(T_i) + sigmav_THe3_D_ENDFB_VIII0(T_i)


@relation(
    name="THe3 reactivity ENDFB-VIII1",
    output="sigmav_THe3",
    tags=("fusion_power",),
)
def sigmav_THe3_ENDFB_VIII1(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total T-He3 reactivity from ENDF/B-VIII.1 tabulated cross sections."""
    return sigmav_THe3_np_ENDFB_VIII1(T_i) + sigmav_THe3_D_ENDFB_VIII1(T_i)


@relation(
    name="THe3 reactivity NRL",
    output="sigmav_THe3",
    inputs=("T_i",),
    tags=("fusion_power",),
)
def sigmav_THe3_NRL(
    T_i: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total T-He3 reactivity from NRL Plasma Formulary tabulated rates."""
    return reactivity_from_reactivity_table(
        "THe3_total_reactivity_NRL.yaml",
        T_i,
        interpolation_kind=interpolation_kind,
    )
