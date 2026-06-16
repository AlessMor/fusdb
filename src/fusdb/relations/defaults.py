"""Fallback/default relations.

Default relations are weak assumptions.  RelationSystem activates them only when
their outputs are demanded, missing, and not supplied by a non-default producer.
They are reported in ``compiler_report['active_default_relations']`` so the user
can see which assumptions were used.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fusdb import relation


@relation(
    name="Default line averaged density from average density",
    tags=("default", "plasma", "confinement", "tokamak", "stellarator", "mirror"),
    outputs="n_la",
)
def default_line_averaged_density_from_average_density(n_avg: float) -> float:
    """Fallback tokamak approximation: line average equals volume average."""
    return n_avg


@relation(
    name="Default ion temperature profile from average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_i",
)
def default_ion_temperature_profile_from_average_temperature(T_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-temperature profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(T_avg), dtype=float)


@relation(
    name="Default electron temperature profile from average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_e",
)
def default_electron_temperature_profile_from_average_temperature(T_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform electron-temperature profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(T_avg), dtype=float)


@relation(
    name="Default ion density profile from average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_i",
)
def default_ion_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform ion-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


@relation(
    name="Default electron density profile from average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_e",
)
def default_electron_density_profile_from_average_density(n_avg: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fallback uniform electron-density profile on the canonical rho grid."""
    return np.full_like(np.asarray(rho, dtype=float), float(n_avg), dtype=float)


@relation(
    name="Default ion temperature average from plasma average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_i_avg",
)
def default_ion_temperature_average_from_plasma_average_temperature(T_avg: float) -> float:
    return T_avg


@relation(
    name="Default electron temperature average from plasma average temperature",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="T_e_avg",
)
def default_electron_temperature_average_from_plasma_average_temperature(T_avg: float) -> float:
    return T_avg


@relation(
    name="Default ion density average from plasma average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_i_avg",
)
def default_ion_density_average_from_plasma_average_density(n_avg: float) -> float:
    return n_avg


@relation(
    name="Default electron density average from plasma average density",
    tags=("default", "plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_e_avg",
)
def default_electron_density_average_from_plasma_average_density(n_avg: float) -> float:
    return n_avg


# No default peaking: the flat profile shape is selected as the canonical
# profile producer through each profile variable's ``default_relation`` in the
# registry, so the parabolic generators (the only consumers of a peaking
# factor) are filtered out unless a reactor opts into them.  A peaking factor is
# therefore used only when a reactor genuinely provides or requests one.


@relation(
    name="Default squareness",
    tags=("default", "geometry", "tokamak", "stellarator", "mirror"),
    outputs="squareness",
)
def default_squareness() -> float:
    """Fallback plasma squareness for shape geometry.

    Used only when no value is supplied.  Zero squareness is the standard
    D-shaped cross-section assumption.
    """
    return 0.0


@relation(
    name="Default equimolar DT fuel fractions",
    tags=("default", "plasma", "composition", "dt"),
    outputs=("f_D", "f_T"),
)
def default_equimolar_dt_fuel_fractions() -> tuple[float, float]:
    """Fallback pure-DT fuel composition: f_D = f_T = 0.5.

    This is only appropriate when the scenario is explicitly a DT case and no
    supplied or non-default composition model provides f_D/f_T or n_D/n_T.
    """
    return 0.5, 0.5


@relation(
    name="Default no minority fuel fractions",
    tags=("default", "plasma", "composition", "dt"),
    outputs=("f_He3", "f_He4", "f_Imp"),
)
def default_no_minority_fuel_fractions() -> tuple[float, float, float]:
    """Fallback no-minority composition for simple pure-DT cases."""
    return 1.0e-10, 1.0e-10, 1.0e-10


@relation(
    name="Default zero advanced fusion powers",
    tags=("default", "fusion_power"),
    outputs=(
        "P_fus_He3He3",
        "P_fus_He3He3_alpha",
        "P_fus_He3He3_p",
        "P_fus_THe3",
        "P_fus_THe3_D",
        "P_fus_THe3_D_alpha",
        "P_fus_THe3_D_D",
        "P_fus_THe3_np",
        "P_fus_THe3_np_alpha",
        "P_fus_THe3_np_n",
        "P_fus_THe3_np_p",
        "P_fus_TT_alpha",
        "P_fus_TT_n",
    ),
)
def default_zero_advanced_fusion_powers() -> tuple[float, ...]:
    """Fallback zero power for omitted non-primary fusion branches.

    Returns:
        Zero power for each advanced fusion-power output.
    """
    return (0.0,) * 13
