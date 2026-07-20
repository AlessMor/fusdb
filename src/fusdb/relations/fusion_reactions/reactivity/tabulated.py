"""Tabulated fusion reactivity relations."""

from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from fusdb.relation import relation
from fusdb.utils.datasets import reactivity_from_reactivity_table
from fusdb.utils.datasets import reactivity_from_xsection_table


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
        "reactivity_NRL_DD-total",
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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_DDn", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_DDp", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_DDn", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_DDp", T_i)


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
        "reactivity_NRL_DHe3",
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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_DHe3", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_DHe3", T_i)


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
        "reactivity_NRL_DT",
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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_DT", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_DT", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_He3He3", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_He3He3", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_THe3D", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_THe3D", T_i)


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
            "reactivity_NRL_THe3-total",
            T_i,
            interpolation_kind=interpolation_kind,
        )
        * 0.43
    )


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_THe3n", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_THe3n", T_i)


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
            "reactivity_NRL_THe3-total",
            T_i,
            interpolation_kind=interpolation_kind,
        )
        * 0.51
    )


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
        "reactivity_NRL_THe3-total",
        T_i,
        interpolation_kind=interpolation_kind,
    )


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII0_TT", T_i)


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
    return reactivity_from_xsection_table("xsection_ENDFB-VIII1_TT", T_i)


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
        "reactivity_NRL_TT",
        T_i,
        interpolation_kind=interpolation_kind,
    )
