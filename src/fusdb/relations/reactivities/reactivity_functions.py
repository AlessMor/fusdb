import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray

from fusdb.registry import KEV_TO_J
from fusdb.relations.reactivities.nrl_reactivity_tables import nrl_tabulated_reactivity
from fusdb.relations.reactivities.tabulated_reactivities import tabulated_reactivity


_AVOGADRO_NUMBER = 6.02214076e23
_BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
_KEV_TO_T9 = KEV_TO_J / _BOLTZMANN_CONSTANT_J_PER_K / 1.0e9


def _math_module(value: object):
    if isinstance(value, sp.Expr):
        return sp
    return np


def _temperature_to_t9(ion_temp_profile: object) -> object:
    """Convert ion temperature from keV to units of 1e9 K."""
    return ion_temp_profile * _KEV_TO_T9


def _zero_like(value: object):
    """Return a zero matching the scalar-vs-array shape of ``value``."""
    return value * 0.0


# TODO(low): add DTHively and DDHively reactivities and options to switch between them and BoschHale
# check ranges of validity for each function

def sigmav_DT_BoschHale(ion_temp_profile: float64) -> float64:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    :func:`sigmav_DT_BoschHale` is more accurate than :func:`sigmav_DT` for ion_temp_profile > ~48.45 keV (estimate based on
    linear interp between errors found at available datapoints).
    Maximum error = 1.4% within range 50-1000 keV from available NRL data.

    Formulation from :cite:`bosch_improved_1992`

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s.

    """

    # Bosch Hale coefficients for DT reaction
    C = [0.0, 1.173e-9, 1.514e-2, 7.519e-2, 4.606e-3, 1.35e-2, -1.068e-4, 1.366e-5]
    B_G = 34.3827
    mr_c2 = 1124656

    theta = ion_temp_profile / (
        1
        - (ion_temp_profile * (C[2] + ion_temp_profile * (C[4] + ion_temp_profile * C[6])))
        / (1 + ion_temp_profile * (C[3] + ion_temp_profile * (C[5] + ion_temp_profile * C[7])))
    )
    eta = (B_G**2 / (4 * theta)) ** (1 / 3)
    math = _math_module(ion_temp_profile)
    sigmav = C[1] * theta * math.sqrt(eta / (mr_c2 * ion_temp_profile**3)) * math.exp(-3 * eta)

    return sigmav * 1e-6  # type: ignore[no-any-return] # [m^3/s]


def sigmav_DD_BoschHale(ion_temp_profile: float64) -> tuple[float64, float64, float64]:
    r"""Deuterium-Deuterium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 3.8% within range 5-50 keV and increases significantly outside of [5, 50] keV.

    Uses DD cross section formulation from :cite:`bosch_improved_1992`.

    Other form in :cite:`langenbrunner_analytic_2017`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` tuple (total, D(d,p)T, D(d,n)3He) in m^3/s.
    """

    # For D(d,n)3He
    cBH_1 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0]
    mc2_1 = 937814.0

    # For D(d,p)T
    cBH_2 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0]
    mc2_2 = 937814.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3)
        )
    )

    thetaBH_2 = ion_temp_profile / (
        1
        - (
            (cBH_2[2] * ion_temp_profile + cBH_2[4] * ion_temp_profile**2 + cBH_2[6] * ion_temp_profile**3)
            / (1 + cBH_2[3] * ion_temp_profile + cBH_2[5] * ion_temp_profile**2 + cBH_2[7] * ion_temp_profile**3)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
    etaBH_2: float = cBH_2[0] / (thetaBH_2 ** (1.0 / 3.0))

    math = _math_module(ion_temp_profile)
    sigmav_DDn: float64 = cBH_1[1] * thetaBH_1 * math.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * math.exp(-3.0 * etaBH_1)
    sigmav_DDp: float64 = cBH_2[1] * thetaBH_2 * math.sqrt(etaBH_2 / (mc2_2 * (ion_temp_profile**3.0))) * math.exp(-3.0 * etaBH_2)
    sigmav_tot: float64 = sigmav_DDn + sigmav_DDp

    # (total, D(d,p)T, D(d,n)3He)
    return sigmav_tot * 1e-6, sigmav_DDn * 1e-6, sigmav_DDp * 1e-6  # [m^3/s]


def sigmav_DHe3_BoschHale(ion_temp_profile: float64) -> float64:
    r"""Deuterium-Helium-3 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 8.4% within range 2-100 keV and should not be used outside range [2, 100] keV.

    Uses DD cross section formulation :cite:`bosch_improved_1992`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in m^3/s.
    """

    # For He3(d,p)4He
    cBH_1 = [
        ((68.7508**2) / 4.0) ** (1.0 / 3.0),
        5.51036e-10,
        6.41918e-03,
        -2.02896e-03,
        -1.91080e-05,
        1.35776e-04,
        0,
        0,
    ]
    mc2_1 = 1124572.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3.0)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

    math = _math_module(ion_temp_profile)
    sigmav: float64 = cBH_1[1] * thetaBH_1 * math.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * math.exp(-3.0 * etaBH_1)

    return sigmav * 1e-6  # [m^3/s]


def sigmav_DD_total_NRL(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total DD reactivity from NRL Plasma Formulary tabulated rates."""
    return nrl_tabulated_reactivity(
        "DD_total_NRL",
        ion_temp_profile,
        interpolation_kind=interpolation_kind,
    )


def sigmav_DT_NRL(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DT reactivity from NRL Plasma Formulary tabulated rates."""
    return nrl_tabulated_reactivity(
        "DT_NRL",
        ion_temp_profile,
        interpolation_kind=interpolation_kind,
    )


def sigmav_DHe3_NRL(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """D-He3 reactivity from NRL Plasma Formulary tabulated rates."""
    return nrl_tabulated_reactivity(
        "DHe3_NRL",
        ion_temp_profile,
        interpolation_kind=interpolation_kind,
    )


def sigmav_DT_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DT reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("DT_ENDFB_VIII0", ion_temp_profile)


def sigmav_DDn_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (He3+n) branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("DDn_ENDFB_VIII0", ion_temp_profile)


def sigmav_DDp_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """DD (T+p) branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("DDp_ENDFB_VIII0", ion_temp_profile)


def sigmav_DD_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> tuple[
    float64 | NDArray[np.float64] | sp.Expr,
    float64 | NDArray[np.float64] | sp.Expr,
    float64 | NDArray[np.float64] | sp.Expr,
]:
    """DD reactivity tuple ``(total, DDn, DDp)`` from ENDF/B-VIII.0 tabulated cross sections."""
    sigmav_ddn = sigmav_DDn_ENDFB_VIII0(ion_temp_profile)
    sigmav_ddp = sigmav_DDp_ENDFB_VIII0(ion_temp_profile)
    return sigmav_ddn + sigmav_ddp, sigmav_ddn, sigmav_ddp


def sigmav_DHe3_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """D-He3 reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("DHe3_ENDFB_VIII0", ion_temp_profile)


def sigmav_TT_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""Tritium-Tritium reaction using the CF88 parametrization."""
    T9 = _temperature_to_t9(ion_temp_profile)
    math = _math_module(T9)
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


def sigmav_TT_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """TT reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("TT_ENDFB_VIII0", ion_temp_profile)


def sigmav_TT_NRL(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """TT reactivity from NRL Plasma Formulary tabulated rates."""
    return nrl_tabulated_reactivity(
        "TT_NRL",
        ion_temp_profile,
        interpolation_kind=interpolation_kind,
    )


def sigmav_He3He3_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""Helium-3/Helium-3 reaction using the CF88 parametrization."""
    T9 = _temperature_to_t9(ion_temp_profile)
    math = _math_module(T9)
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


def sigmav_He3He3_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """He3-He3 reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("He3He3_ENDFB_VIII0", ion_temp_profile)


def sigmav_THe3_D_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""T + 3He -> 4He + D branch using the CF88 parametrization."""
    T9 = _temperature_to_t9(ion_temp_profile)
    T9A = T9 / (1 + 0.128 * T9)
    math = _math_module(T9)
    # The CF88 shorthand is T9A^(5/6) * exp(-7.733 / T9A^(1/3)).
    sigmav = 5.46e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * math.exp(-7.733 / (T9A ** (1 / 3)))
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


def sigmav_THe3_D_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + D branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("THe3D_ENDFB_VIII0", ion_temp_profile)


def sigmav_THe3_np_CF88(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""T + 3He -> 4He + n + p branch using the CF88 parametrization."""
    T9 = _temperature_to_t9(ion_temp_profile)
    T9A = T9 / (1 + 0.115 * T9)
    math = _math_module(T9)
    # The CF88 shorthand is T9A^(5/6) * exp(-7.733 / T9A^(1/3)).
    sigmav = 7.71e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * math.exp(-7.733 / (T9A ** (1 / 3)))
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


def sigmav_THe3_np_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """T + He3 -> alpha + n + p branch reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return tabulated_reactivity("THe3n_ENDFB_VIII0", ion_temp_profile)


def sigmav_THe3_He5p(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    r"""T + 3He -> 5He + p placeholder channel.

    This branch keeps the higher-level API stable while a validated parametrization is still missing.
    """
    return _zero_like(ion_temp_profile)  # type: ignore[no-any-return]


def sigmav_THe3_CF88(
    ion_temp_profile: float64 | NDArray[np.float64],
) -> tuple[
    float64 | NDArray[np.float64],
    float64 | NDArray[np.float64],
    float64 | NDArray[np.float64],
]:
    r"""Return the three T + 3He CF88 branch reactivities.

    The tuple order is ``(alpha+n+p, alpha+D, He5+p)``.
    """
    ch1 = sigmav_THe3_np_CF88(ion_temp_profile)
    ch2 = sigmav_THe3_D_CF88(ion_temp_profile)
    ch3 = sigmav_THe3_He5p(ion_temp_profile)
    return ch1, ch2, ch3


def sigmav_TT(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    """Return the placeholder default T-T reactivity."""
    return _zero_like(ion_temp_profile)  # type: ignore[no-any-return]


def sigmav_He3He3(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    """Return the placeholder default He3-He3 reactivity."""
    return _zero_like(ion_temp_profile)  # type: ignore[no-any-return]


def sigmav_THe3(ion_temp_profile: float64 | NDArray[np.float64]) -> float64 | NDArray[np.float64]:
    """Return the placeholder default T-He3 reactivity."""
    return _zero_like(ion_temp_profile)  # type: ignore[no-any-return]


def sigmav_THe3_ENDFB_VIII0(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total T-He3 reactivity from ENDF/B-VIII.0 tabulated cross sections."""
    return sigmav_THe3_np_ENDFB_VIII0(ion_temp_profile) + sigmav_THe3_D_ENDFB_VIII0(ion_temp_profile)


def sigmav_THe3_NRL(
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
    *,
    interpolation_kind: str = "pchip",
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Total T-He3 reactivity from NRL Plasma Formulary tabulated rates."""
    return nrl_tabulated_reactivity(
        "THe3_total_NRL",
        ion_temp_profile,
        interpolation_kind=interpolation_kind,
    )
