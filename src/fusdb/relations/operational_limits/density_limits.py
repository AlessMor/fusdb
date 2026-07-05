"""Operational density limit relations."""

from typing import Any

import numpy as np

from fusdb import relation


@relation(
    name='Greenwald density limit',
    tags=('plasma', 'tokamak'),
    
    outputs='n_GW',
)
def greenwald_density_limit(I_p: float, a: float) -> Any:
    """Return Greenwald density limit in 1/m^3 for tokamaks."""
    I_p_MA = I_p / 1e6
    return 1e20 * I_p_MA / (np.pi * a**2)


@relation(
    name='Greenwald density fraction',
    tags=('plasma', 'tokamak'),
    
    outputs='f_GW',
)
def greenwald_density_fraction(n_GW: float, n_avg: float) -> Any:
    """Return fraction of Greenwald density limit."""
    f_GW =  n_avg / n_GW
    return f_GW


@relation(
    name='Greenwald margin',
    tags=('plasma', 'tokamak', 'constraint'),
    outputs='greenwald_margin',
)
def greenwald_margin(n_avg: float, n_GW: float) -> Any:
    """Return Greenwald margin (<=0 satisfied)."""
    return n_avg - n_GW


@relation(
    name="Edge perpendicular power density",
    tags=("power_exhaust", "process"),
    outputs="p_perp_edge",
)
def edge_perpendicular_power_density(P_sep: float, a_plasma_surface: float) -> Any:
    """Return the power per unit area crossing the plasma edge (excludes
    radiation and neutrons).

    Adapted from PROCESS; see README.md section "Third-party Notices".

    PROCESS computes this intermediate (``p_perp``) inside its density-limit
    dispatcher in MW/m^2; fusdb stores it in SI (W/m^2).
    """
    # CHECK
    return P_sep / a_plasma_surface


@relation(
    name="Density limit ASDEX",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_asdex_density_limit(
    p_perp_edge: float, b_plasma_toroidal_on_axis: float, q95: float, rmajor: float, nesep_over_nebar: float
) -> Any:
    """Calculate the (old) ASDEX density limit, scaled from the edge to the
    average plasma density by the edge/average density ratio.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - T.C.Hender et.al., 'Physics Assessment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # CHECK
    p_perp = p_perp_edge / 1.0e6  # PROCESS uses MW/m^2
    return (
        1.54e20
        * p_perp**0.43
        * b_plasma_toroidal_on_axis**0.31
        / (q95 * rmajor) ** 0.45
    ) / nesep_over_nebar


@relation(
    name="Density limit Borrass ITER I",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_borrass_iter_i_density_limit(
    p_perp_edge: float, b_plasma_toroidal_on_axis: float, q95: float, rmajor: float, nesep_over_nebar: float
) -> Any:
    """Calculate the Borrass ITER I density limit, scaled from the edge to the
    average plasma density by the edge/average density ratio.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - T.C.Hender et.al., 'Physics Assessment of the European Reactor Study',
          AEA FUS 172, 1992
        - Borrass et al, ITER-TN-PH-9-6 (1989)
    """
    # CHECK
    p_perp = p_perp_edge / 1.0e6  # PROCESS uses MW/m^2
    return (
        1.8e20
        * p_perp**0.53
        * b_plasma_toroidal_on_axis**0.31
        / (q95 * rmajor) ** 0.22
    ) / nesep_over_nebar


@relation(
    name="Density limit Borrass ITER II",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_borrass_iter_ii_density_limit(
    p_perp_edge: float, b_plasma_toroidal_on_axis: float, q95: float, rmajor: float, nesep_over_nebar: float
) -> Any:
    """Calculate the Borrass ITER II density limit, scaled from the edge to the
    average plasma density by the edge/average density ratio.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - T.C.Hender et.al., 'Physics Assessment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # CHECK
    p_perp = p_perp_edge / 1.0e6  # PROCESS uses MW/m^2
    return (
        0.5e20
        * p_perp**0.57
        * b_plasma_toroidal_on_axis**0.31
        / (q95 * rmajor) ** 0.09
    ) / nesep_over_nebar


@relation(
    name="Density limit JET edge radiation",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_jet_edge_radiation_density_limit(
    Z_eff: float, P_aux: float, nesep_over_nebar: float, qstar: float
) -> Any:
    """Calculate the JET edge radiation density limit, scaled from the edge to
    the average plasma density by the edge/average density ratio.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``qstar`` is the equivalent cylindrical safety factor (PROCESS ``qcyl``);
    ``P_aux`` is the injected heating power (PROCESS p_hcd_injected_total_mw).

    References
    ----------
        - T.C.Hender et.al., 'Physics Assessment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # CHECK
    p_hcd_injected_total_mw = P_aux / 1.0e6  # PROCESS uses MW
    denom = (Z_eff - 1.0) * (1.0 - 4.0 / (3.0 * qstar))
    if denom <= 0.0:
        return 0.0
    return (1.0e20 * np.sqrt(p_hcd_injected_total_mw / denom)) / nesep_over_nebar


@relation(
    name="Density limit JET simple",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_jet_simple_density_limit(
    b_plasma_toroidal_on_axis: float, P_sep: float, rmajor: float, nesep_over_nebar: float
) -> Any:
    """Calculate the JET simplified density limit, scaled from the edge to the
    average plasma density by the edge/average density ratio.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - T.C.Hender et.al., 'Physics Assessment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # CHECK
    p_plasma_separatrix_mw = P_sep / 1.0e6  # PROCESS uses MW
    return (
        0.237e20
        * b_plasma_toroidal_on_axis
        * np.sqrt(p_plasma_separatrix_mw)
        / rmajor
    ) / nesep_over_nebar


@relation(
    name="Density limit Hugill-Murakami",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_hugill_murakami_density_limit(
    b_plasma_toroidal_on_axis: float, rmajor: float, qstar: float
) -> Any:
    """Calculate the Hugill-Murakami M.q density limit.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``qstar`` is the equivalent cylindrical safety factor (PROCESS ``qcyl``).

    References
    ----------
        - N.A. Uckan and ITER Physics Group, 'ITER Physics Design Guidelines: 1989'
    """
    # CHECK
    return 3.0e20 * b_plasma_toroidal_on_axis / (rmajor * qstar)


@relation(
    name="Density limit ASDEX New",
    tags=("plasma", "tokamak", "process"),
    outputs="n_e_max",
)
def calculate_asdex_new_density_limit(
    P_aux: float, plasma_current: float, q95: float, nesep_over_nebar: float
) -> Any:
    """Calculate the ASDEX Upgrade new (H-mode) density limit, scaled from the
    separatrix to the average plasma density by the edge/average density ratio.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    References
    ----------
        - J. W. Berkery et al., "Density limits as disruption forecasters for
          spherical tokamaks," PPCF 65, 095003 (2023)
        - M. Bernert et al., "The H-mode density limit in the full tungsten
          ASDEX Upgrade tokamak," PPCF 57, 014038 (2014)
    """
    # CHECK
    p_hcd_injected_total_mw = P_aux / 1.0e6  # PROCESS uses MW
    return (
        1.0e20
        * 0.506
        * (p_hcd_injected_total_mw**0.396 * (plasma_current / 1.0e6) ** 0.265)
        / (q95**0.323)
    ) / nesep_over_nebar


@relation(
    name='Sudo density limit',
    tags=('plasma', 'stellarator'),

    outputs='n_SUDO',
)
def sudo_density_limit(P_loss: float, B0: float, R: float, a: float) -> Any:
    """Return Sudo density limit in 1/m^3 for stellarators."""
    P_loss_MW = P_loss / 1e6
    return 1e20 * 0.25 * P_loss_MW * B0 / (R * a**2)


@relation(
    name='Sudo margin',
    tags=('plasma', 'stellarator', 'constraint'),
    outputs='sudo_margin',
)
def sudo_margin(n_avg: float, n_SUDO: float) -> Any:
    """Return Sudo margin (<=0 satisfied)."""
    return n_avg - n_SUDO
