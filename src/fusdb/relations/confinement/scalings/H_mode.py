"""H-mode confinement scaling relations."""

import numpy as np

from fusdb.relation import relation


@relation(
    name="mirnov_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def mirnov_confinement_time(rminor: float, kappa95: float, I_p: float) -> float:
    """Calculate the Mirnov scaling (H-mode) confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rminor :
        Plasma minor radius [m]
    kappa95 :
        Plasma elongation at 95% flux surface
    pcur :
        Plasma current [MA]

    Returns
    -------
    :
        float: Mirnov scaling confinement time [s]

    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria) and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    return 0.2e0 * rminor * np.sqrt(kappa95) * pcur


@relation(
    name="murari_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def murari_confinement_time(
    I_p: float,
    rmajor: float,
    kappa_ipb: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Murari H-mode energy confinement scaling time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    rmajor :
        Plasma major radius [m]
    kappa_ipb :
        IPB specific plasma separatrix elongation
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Murari confinement time [s]

    Notes
    -----
        - This scaling uses the IPB defintiion of elongation, see reference for
          more information.

    References
    ----------
        - A. Murari, E. Peluso, Michela Gelfusa, I. Lupelli, and P. Gaudio,
         “A new approach to the formulation and validation of scaling expressions
         for plasma confinement in tokamaks,” Nuclear Fusion, vol. 55, no. 7,
         pp. 073009-073009, Jun. 2015,
         doi: https://doi.org/10.1088/0029-5515/55/7/073009.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy,
          “Corrections to a sequence of papers in Nuclear Fusion,” Nuclear Fusion,
          vol. 48, no. 9, pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0367
        * pcur**1.006
        * rmajor**1.731
        * kappa_ipb**1.450
        * p_plasma_loss_mw ** (-0.735)
        * (
            dnla19**0.448
            / (1.0 + np.exp(-9.403 * (dnla19 / b_plasma_toroidal_on_axis) ** -1.365))
        )
    )


@relation(
    name="shimomura_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def shimomura_confinement_time(
    rmajor: float,
    rminor: float,
    b_plasma_toroidal_on_axis: float,
    kappa95: float,
    afuel: float,
) -> float:
    """Calculate the  Shimomura (S) optimized H-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    kappa95 :
        Plasma elongation at 95% flux surface
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: Shimomura confinement time [s]

    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    return (
        0.045e0
        * rmajor
        * rminor
        * b_plasma_toroidal_on_axis
        * np.sqrt(kappa95)
        * np.sqrt(afuel)
    )

@relation(
    name="riedel_h_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def riedel_h_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa95: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Riedel scaling (H-mode) confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa95 :
        Plasma elongation at 95% flux surface
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    afuel :
        Fuel atomic mass number
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Riedel H-mode confinement time [s]

    References
    ----------
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.1e0
        * np.sqrt(afuel)
        * pcur**0.884e0
        * rmajor**1.24e0
        * rminor ** (-0.23e0)
        * kappa95**0.317e0
        * b_plasma_toroidal_on_axis**0.207e0
        * dnla20**0.105e0
        / p_plasma_loss_mw**0.486e0
    )


@relation(
    name="valovic_elmy_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def valovic_elmy_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    afuel: float,
    rmajor: float,
    rminor: float,
    kappa: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Valovic modified ELMy-H mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    afuel :
        Fuel atomic mass number
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa :
        Plasma elongation
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Valovic modified ELMy-H mode confinement time [s]
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.067e0
        * pcur**0.9e0
        * b_plasma_toroidal_on_axis**0.17e0
        * dnla19**0.45e0
        * afuel**0.05e0
        * rmajor**1.316e0
        * rminor**0.79e0
        * kappa**0.56e0
        * p_plasma_loss_mw ** (-0.68e0)
    )



@relation(
    name="lang_high_density_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def lang_high_density_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    rminor: float,
    q95: float,
    qstar: float,
    aspect: float,
    afuel: float,
    kappa_ipb: float,
) -> float:
    """Calculate the high density relevant confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    plasma_current :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    nd_plasma_electron_line :
        Line averaged electron density [m**-3]
    p_plasma_loss_mw :
        Net Heating power [MW]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    q :
        Safety factor
    qstar :
        Equivalent cylindrical edge safety factor
    aspect :
        Aspect ratio
    afuel :
        Fuel atomic mass number
    kappa_ipb :
        Plasma elongation at 95% flux surface

    Returns
    -------
    :
        float: High density relevant confinement time [s]

    References
    ----------
        - P. T. Lang, C. Angioni, R. M. M. Dermott, R. Fischer, and H. Zohm,
          “Pellet Induced High Density Phases during ELM Suppression in
          ASDEX Upgrade,” 24th IAEA Conference Fusion Energy, 2012, Oct. 2012,
          Available: https://www.researchgate.net/publication/274456104_Pellet_Induced_High_Density_Phases_during_ELM_Suppression_in_ASDEX_Upgrade
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    plasma_current = I_p
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    nd_plasma_electron_line = n_la
    q = q95
    qratio = q / qstar
    n_gw = 1.0e14 * plasma_current / (np.pi * rminor * rminor)
    nratio = nd_plasma_electron_line / n_gw
    return (
        6.94e-7
        * plasma_current**1.3678e0
        * b_plasma_toroidal_on_axis**0.12e0
        * nd_plasma_electron_line**0.032236e0
        * (p_plasma_loss_mw * 1.0e6) ** (-0.74e0)
        * rmajor**1.2345e0
        * kappa_ipb**0.37e0
        * aspect**2.48205e0
        * afuel**0.2e0
        * qratio**0.77e0
        * aspect ** (-0.9e0 * np.log(aspect))
        * nratio ** (-0.22e0 * np.log(nratio))
    )

@relation(
    name="cfspopcon_h_ds03_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E_scaling",
)
def cfspopcon_h_ds03_confinement_time(
    I_p: float,
    B0: float,
    n_avg: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    afuel: float,
) -> float:
    """Calculate the H_DS03 confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - Electrostatic, GyroBohm-like confinement scaling, eqn 21 from Petty
          et al 'Feasibility Study of a Compact Ignition Tokamak Based Upon
          GyroBohm Scaling Physics.' (2003), Fusion Science and Technology, 43

    Notes
    -----
        - inverse_aspect_ratio_alpha note: (major_radius/a)^-0.3 =
          (a/major_radius)^0.3
        - mass_ratio_alpha note: a_M, isotope mass scaling
        - Regime: H-mode
        - Elongation: cfspopcon puts this scaling's 0.75 exponent on
          ``separatrix_elongation_alpha``, NOT ``areal_elongation_alpha``
          (energy_confinement_scalings.yaml, H_DS03).  It therefore takes
          fusdb's ``kappa``, which IS the separatrix elongation
          (kappa == kappa_sep == kappa_geom at psi_N = 1).  Do not "correct"
          this to ``kappa_separatrix``: that variable is redundant with
          ``kappa`` and is scheduled to be merged into it.
    """
    return (
        0.028
        * (I_p / 1.0e6) ** 0.83
        * B0**0.07
        * (n_avg / 1.0e19) ** 0.49
        * (P_loss / 1.0e6) ** (-0.55)
        * R**2.11
        * eps**0.3
        * kappa**0.75
        * afuel**0.14
    )
