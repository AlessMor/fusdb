"""Extended two-point model (Stangeby PPCF 2018) for divertor target conditions.

cfspopcon solves the coupled (separatrix temp, target temp/density/flux) system
iteratively; here the closed-form component relations are imported and fusdb's
reconcile solves the coupling as a zero-residual system. The iterative drivers
(``solve_two_point_model``, ``two_point_model_fixed_*``, target-first model) are
not imported.

cfspopcon works in eV/atm/GW; the formula constants are pure in SI, so these are
computed in SI with a keV<->J conversion (``KEV_TO_J``) where temperature appears.
The Spitzer-Harm ``kappa_e0`` is eV-based, so only ``separatrix_electron_temp``
and the momentum-loss fits use an explicit keV->eV factor.
"""

from typing import Any

import numpy as np

from fusdb import relation
from fusdb.registry import ATOMIC_MASS_UNIT_KG, KEV_TO_J

_KEV_TO_EV = 1.0e3


@relation(
    name="Separatrix electron temperature (Spitzer-Harm)",
    tags=("power_exhaust", "tokamak"),
    outputs="T_sep",
)
def calc_separatrix_electron_temp(
    target_electron_temp: Any,
    q_parallel: Any,
    parallel_connection_length: Any,
    kappa_e0: Any,
    SOL_conduction_fraction: Any = 1.0,
) -> Any:
    """Upstream electron temperature assuming Spitzer-Harm heat conductivity.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 38 from :cite:`stangeby_2018`. ``kappa_e0`` is eV-based, so temperatures
    are handled in eV here and the result converted back to keV.

    Args:
        target_electron_temp: [keV] :term:`glossary link<target_electron_temp>`
        q_parallel: [W/m^2] :term:`glossary link<q_parallel>`
        parallel_connection_length: [m] :term:`glossary link<parallel_connection_length>`
        kappa_e0: [W/(eV^3.5 m)] :term:`glossary link<kappa_e0>`
        SOL_conduction_fraction: [~] :term:`glossary link<SOL_conduction_fraction>`

    Returns:
        T_sep [keV]
    """
    # CHECK
    target_eV = target_electron_temp * _KEV_TO_EV
    upstream_eV = (target_eV**3.5 + 3.5 * (SOL_conduction_fraction * q_parallel * parallel_connection_length / kappa_e0)) ** (2.0 / 7.0)
    return upstream_eV / _KEV_TO_EV


@relation(
    name="Upstream total pressure",
    tags=("power_exhaust", "tokamak"),
    outputs="upstream_total_pressure",
)
def calc_upstream_total_pressure(
    n_sep: Any,
    T_sep: Any,
    upstream_ratio_of_ion_to_electron_temp: Any = 1.0,
    upstream_ratio_of_electron_to_ion_density: Any = 1.0,
    upstream_mach_number: Any = 0.0,
) -> Any:
    """Total (electron + ion) upstream SOL pressure (Stangeby 2018 eq. 20).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        (1.0 + upstream_mach_number**2)
        * n_sep
        * (T_sep * KEV_TO_J)
        * (1.0 + upstream_ratio_of_ion_to_electron_temp / upstream_ratio_of_electron_to_ion_density)
    )


@relation(
    name="Target electron temperature (basic 2PM)",
    tags=("power_exhaust", "tokamak"),
    outputs="target_electron_temp_basic",
)
def calc_target_electron_temp_basic(afuel: Any, q_parallel: Any, upstream_total_pressure: Any, sheath_heat_transmission_factor: Any) -> Any:
    """Target electron temperature from the basic two-point model (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    m_i = afuel * ATOMIC_MASS_UNIT_KG
    target_J = (8.0 * m_i / sheath_heat_transmission_factor**2) * (q_parallel**2 / upstream_total_pressure**2)
    return target_J / KEV_TO_J


@relation(
    name="Target electron density (basic 2PM)",
    tags=("power_exhaust", "tokamak"),
    outputs="target_electron_density_basic",
)
def calc_target_electron_density_basic(afuel: Any, q_parallel: Any, upstream_total_pressure: Any, sheath_heat_transmission_factor: Any) -> Any:
    """Target electron density from the basic two-point model (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    m_i = afuel * ATOMIC_MASS_UNIT_KG
    return sheath_heat_transmission_factor**2 / (32.0 * m_i) * upstream_total_pressure**3 / q_parallel**2


@relation(
    name="Target electron flux (basic 2PM)",
    tags=("power_exhaust", "tokamak"),
    outputs="target_electron_flux_basic",
)
def calc_target_electron_flux_basic(afuel: Any, q_parallel: Any, upstream_total_pressure: Any, sheath_heat_transmission_factor: Any) -> Any:
    """Target electron flux from the basic two-point model (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    m_i = afuel * ATOMIC_MASS_UNIT_KG
    return sheath_heat_transmission_factor / (8.0 * m_i) * upstream_total_pressure**2 / q_parallel


@relation(
    name="Volume-loss factor for target electron temperature",
    tags=("power_exhaust", "tokamak"),
    outputs="f_vol_loss_target_electron_temp",
)
def calc_f_vol_loss_target_electron_temp(SOL_power_loss_fraction: Any, SOL_momentum_loss_fraction: Any) -> Any:
    """Volume-loss correction for target electron temperature (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (1.0 - SOL_power_loss_fraction) ** 2 / (1.0 - SOL_momentum_loss_fraction) ** 2


@relation(
    name="Volume-loss factor for target electron density",
    tags=("power_exhaust", "tokamak"),
    outputs="f_vol_loss_target_electron_density",
)
def calc_f_vol_loss_target_electron_density(SOL_power_loss_fraction: Any, SOL_momentum_loss_fraction: Any) -> Any:
    """Volume-loss correction for target electron density (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (1.0 - SOL_momentum_loss_fraction) ** 3 / (1.0 - SOL_power_loss_fraction) ** 2


@relation(
    name="Volume-loss factor for target electron flux",
    tags=("power_exhaust", "tokamak"),
    outputs="f_vol_loss_target_electron_flux",
)
def calc_f_vol_loss_target_electron_flux(SOL_power_loss_fraction: Any, SOL_momentum_loss_fraction: Any) -> Any:
    """Volume-loss correction for target electron flux (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (1.0 - SOL_momentum_loss_fraction) ** 2 / (1.0 - SOL_power_loss_fraction)


@relation(
    name="Other factor for target electron temperature",
    tags=("power_exhaust", "tokamak"),
    outputs="f_other_target_electron_temp",
)
def calc_f_other_target_electron_temp(
    target_ratio_of_ion_to_electron_temp: Any,
    target_ratio_of_electron_to_ion_density: Any,
    target_mach_number: Any,
    toroidal_flux_expansion: Any,
) -> Any:
    """Non-volume-loss correction for target electron temperature (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        ((1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density) / 2.0)
        * ((1.0 + target_mach_number**2) ** 2 / (4.0 * target_mach_number**2))
        * toroidal_flux_expansion**-2
    )


@relation(
    name="Other factor for target electron density",
    tags=("power_exhaust", "tokamak"),
    outputs="f_other_target_electron_density",
)
def calc_f_other_target_electron_density(
    target_ratio_of_ion_to_electron_temp: Any,
    target_ratio_of_electron_to_ion_density: Any,
    target_mach_number: Any,
    toroidal_flux_expansion: Any,
) -> Any:
    """Non-volume-loss correction for target electron density (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        (4.0 / (1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density) ** 2)
        * 8.0
        * target_mach_number**2
        / (1.0 + target_mach_number**2) ** 3
        * toroidal_flux_expansion**2
    )


@relation(
    name="Other factor for target electron flux",
    tags=("power_exhaust", "tokamak"),
    outputs="f_other_target_electron_flux",
)
def calc_f_other_target_electron_flux(
    target_ratio_of_ion_to_electron_temp: Any,
    target_ratio_of_electron_to_ion_density: Any,
    target_mach_number: Any,
    toroidal_flux_expansion: Any,
) -> Any:
    """Non-volume-loss correction for target electron flux (Stangeby 2018 eq. 24).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return (
        (2.0 / (1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density))
        * 4.0
        * target_mach_number**2
        / (1.0 + target_mach_number**2) ** 2
        * toroidal_flux_expansion
    )


@relation(
    name="Target electron temperature",
    tags=("power_exhaust", "tokamak"),
    outputs="target_electron_temp",
)
def calc_target_electron_temp(
    target_electron_temp_basic: Any, f_vol_loss_target_electron_temp: Any, f_other_target_electron_temp: Any
) -> Any:
    """Target electron temperature with volume-loss and other corrections.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return target_electron_temp_basic * f_vol_loss_target_electron_temp * f_other_target_electron_temp


@relation(
    name="Target electron density",
    tags=("power_exhaust", "tokamak"),
    outputs="target_electron_density",
)
def calc_target_electron_density(
    target_electron_density_basic: Any, f_vol_loss_target_electron_density: Any, f_other_target_electron_density: Any
) -> Any:
    """Target electron density with volume-loss and other corrections.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return target_electron_density_basic * f_vol_loss_target_electron_density * f_other_target_electron_density


@relation(
    name="Target electron flux",
    tags=("power_exhaust", "tokamak"),
    outputs="target_electron_flux",
)
def calc_target_electron_flux(
    target_electron_flux_basic: Any, f_vol_loss_target_electron_flux: Any, f_other_target_electron_flux: Any
) -> Any:
    """Target electron flux with volume-loss and other corrections.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return target_electron_flux_basic * f_vol_loss_target_electron_flux * f_other_target_electron_flux


@relation(
    name="Target parallel heat flux from power loss",
    tags=("power_exhaust", "tokamak"),
    outputs="target_q_parallel",
)
def calc_target_q_parallel(q_parallel: Any, SOL_power_loss_fraction: Any) -> Any:
    """Parallel heat-flux density reaching the target after SOL power loss.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return q_parallel * (1.0 - SOL_power_loss_fraction)


@relation(
    name="Required SOL power loss fraction",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_power_loss_fraction",
)
def calc_required_SOL_power_loss_fraction(
    target_electron_temp_basic: Any,
    f_other_target_electron_temp: Any,
    SOL_momentum_loss_fraction: Any,
    required_target_electron_temp: Any,
) -> Any:
    """SOL power loss (cooling) fraction required to reach a desired target electron temperature.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 15 of :cite:`stangeby_2018`, rearranged for f_cooling. Both target
    temperatures are in keV, so their ratio needs no unit conversion.
    """
    # CHECK
    required = 1.0 - np.sqrt(
        required_target_electron_temp
        / target_electron_temp_basic
        * (1.0 - SOL_momentum_loss_fraction) ** 2
        / f_other_target_electron_temp
    )
    return np.maximum(required, 0.0)


def _momentum_loss(A: float, Tstar: float, n: float, target_electron_temp: Any) -> Any:
    """Generic SOL momentum-loss fraction (Stangeby 2018 eq. 33). target_electron_temp in keV."""
    target_eV = target_electron_temp * _KEV_TO_EV
    return 1.0 - A * (1.0 - np.exp(-target_eV / Tstar)) ** n


@relation(
    name="SOL momentum loss fraction KotovReiter",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_momentum_loss_fraction",
)
def calc_SOL_momentum_loss_fraction_KotovReiter(target_electron_temp: Any) -> Any:
    """SOL momentum-loss fraction, Kotov-Reiter fit (Stangeby 2018 fig. 7a).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return _momentum_loss(1.0, 0.8, 2.1, target_electron_temp)


@relation(
    name="SOL momentum loss fraction Sang",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_momentum_loss_fraction",
)
def calc_SOL_momentum_loss_fraction_Sang(target_electron_temp: Any) -> Any:
    """SOL momentum-loss fraction, Sang fit (Stangeby 2018 fig. 7b).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return _momentum_loss(1.3, 1.8, 1.6, target_electron_temp)


@relation(
    name="SOL momentum loss fraction Jarvinen",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_momentum_loss_fraction",
)
def calc_SOL_momentum_loss_fraction_Jarvinen(target_electron_temp: Any) -> Any:
    """SOL momentum-loss fraction, Jarvinen fit (Stangeby 2018 fig. 10a).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return _momentum_loss(1.7, 2.2, 1.2, target_electron_temp)


@relation(
    name="SOL momentum loss fraction Moulton",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_momentum_loss_fraction",
)
def calc_SOL_momentum_loss_fraction_Moulton(target_electron_temp: Any) -> Any:
    """SOL momentum-loss fraction, Moulton fit (Stangeby 2018 fig. 10b).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return _momentum_loss(1.0, 1.0, 1.5, target_electron_temp)


@relation(
    name="SOL momentum loss fraction PerezH",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_momentum_loss_fraction",
)
def calc_SOL_momentum_loss_fraction_PerezH(target_electron_temp: Any) -> Any:
    """SOL momentum-loss fraction, Perez H-mode fit (Stangeby 2018 fig. 11a).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return _momentum_loss(0.8, 2.0, 1.2, target_electron_temp)


@relation(
    name="SOL momentum loss fraction PerezL",
    tags=("power_exhaust", "tokamak"),
    outputs="SOL_momentum_loss_fraction",
)
def calc_SOL_momentum_loss_fraction_PerezL(target_electron_temp: Any) -> Any:
    """SOL momentum-loss fraction, Perez L-mode fit (Stangeby 2018 fig. 11b).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return _momentum_loss(1.1, 3.0, 0.9, target_electron_temp)
