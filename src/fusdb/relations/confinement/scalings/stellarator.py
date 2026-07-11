"""Stellarator confinement scaling relations."""

from fusdb.relation import relation


@relation(
    name="sudo_et_al_confinement_time",
    tags=("confinement", "stellarator"),
    outputs="tau_E",
)
def sudo_et_al_confinement_time(
    rmajor: float,
    rminor: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Sudo et al. scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Sudo et al. confinement time [s]

    References
    ----------
        - S. Sudo et al., “Scalings of energy confinement and density limit in
          stellarator/heliotron devices,” Nuclear Fusion, vol. 30, no. 1,
          pp. 11-21, Jan. 1990, doi: https://doi.org/10.1088/0029-5515/30/1/002.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.17e0
        * rmajor**0.75e0
        * rminor**2
        * dnla20**0.69e0
        * b_plasma_toroidal_on_axis**0.84e0
        * p_plasma_loss_mw ** (-0.58e0)
    )


@relation(
    name="gyro_reduced_bohm_confinement_time",
    tags=("confinement", "stellarator"),
    outputs="tau_E",
)
def gyro_reduced_bohm_confinement_time(
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rminor: float,
    rmajor: float,
) -> float:
    """Calculate the Gyro-reduced Bohm scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    p_plasma_loss_mw :
        Net Heating power [MW]
    rminor :
        Plasma minor radius [m]
    rmajor :
        Plasma major radius [m]

    Returns
    -------
    :
        float: Gyro-reduced Bohm confinement time [s]

    References
    ----------
        - Goldston, R. J., H. Biglari, and G. W. Hammett. "E x B/B 2 vs. μ B/B as
          the Cause of Transport in Tokamaks." Bull. Am. Phys. Soc 34 (1989): 1964.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.25e0
        * b_plasma_toroidal_on_axis**0.8e0
        * dnla20**0.6e0
        * p_plasma_loss_mw ** (-0.6e0)
        * rminor**2.4e0
        * rmajor**0.6e0
    )


@relation(
    name="lackner_gottardi_stellarator_confinement_time",
    tags=("confinement", "stellarator"),
    outputs="tau_E",
)
def lackner_gottardi_stellarator_confinement_time(
    rmajor: float,
    rminor: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    q95: float,
) -> float:
    """Calculate the Lackner-Gottardi stellarator scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    q :
        Edge safety factor

    Returns
    -------
    :
        float: Lackner-Gottardi stellarator confinement time [s]

    References
    ----------
        - K. Lackner and N. A. O. Gottardi, “Tokamak confinement in relation to
          plateau scaling,” Nuclear Fusion, vol. 30, no. 4, pp. 767-770, Apr. 1990,
          doi: https://doi.org/10.1088/0029-5515/30/4/018.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    q = q95
    return (
        0.17e0
        * rmajor
        * rminor**2
        * dnla20**0.6e0
        * b_plasma_toroidal_on_axis**0.8e0
        * p_plasma_loss_mw ** (-0.6e0)
        * q**0.4e0
    )


@relation(
    name="iss95_stellarator_confinement_time",
    tags=("confinement", "stellarator"),
    outputs="tau_E",
)
def iss95_stellarator_confinement_time(
    rminor: float,
    rmajor: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    iotabar: float,
) -> float:
    """Calculate the ISS95 stellarator scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rminor :
        Plasma minor radius [m]
    rmajor :
        Plasma major radius [m]
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    iotabar :
        Rotational transform

    Returns
    -------
    :
        float: ISS95 stellarator confinement time [s]

    References
    ----------
        - U. Stroth et al., “Energy confinement scaling from the international
          stellarator database,” vol. 36, no. 8, pp. 1063-1077, Aug. 1996,
          doi: https://doi.org/10.1088/0029-5515/36/8/i11.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.079e0
        * rminor**2.21e0
        * rmajor**0.65e0
        * dnla19**0.51e0
        * b_plasma_toroidal_on_axis**0.83e0
        * p_plasma_loss_mw ** (-0.59e0)
        * iotabar**0.4e0
    )


@relation(
    name="iss04_stellarator_confinement_time",
    tags=("confinement", "stellarator"),
    outputs="tau_E",
)
def iss04_stellarator_confinement_time(
    rminor: float,
    rmajor: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    iotabar: float,
) -> float:
    """Calculate the ISS04 stellarator scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rminor :
        Plasma minor radius [m]
    rmajor :
        Plasma major radius [m]
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    iotabar :
        Rotational transform

    Returns
    -------
    :
        float: ISS04 stellarator confinement time [s]

    References
    ----------
        - H. Yamada et al., “Characterization of energy confinement in net-current
          free plasmas using the extended International Stellarator Database,”
          vol. 45, no. 12, pp. 1684-1693, Nov. 2005,
          doi: https://doi.org/10.1088/0029-5515/45/12/024.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.134e0
        * rminor**2.28e0
        * rmajor**0.64e0
        * dnla19**0.54e0
        * b_plasma_toroidal_on_axis**0.84e0
        * p_plasma_loss_mw ** (-0.61e0)
        * iotabar**0.41e0
    )


@relation(
    name="ds03_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def ds03_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa95: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the DS03 beta-independent H-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw :
        Net Heating power [MW]
    rmajor :
        Plasma major radius [m]
    kappa95 :
        Plasma elongation at 95% flux surface
    aspect :
        Aspect ratio
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: DS03 beta-independent H-mode confinement time [s]

    References
    ----------
        - T. C. Luce, C. C. Petty, and J. G. Cordey, “Application of dimensionless
          parameter scaling techniques to the design and interpretation of magnetic
          fusion experiments,” Plasma Physics and Controlled Fusion, vol. 50, no. 4,
          p. 043001, Mar. 2008, doi: https://doi.org/10.1088/0741-3335/50/4/043001.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.028e0
        * pcur**0.83e0
        * b_plasma_toroidal_on_axis**0.07e0
        * dnla19**0.49e0
        * p_plasma_loss_mw ** (-0.55e0)
        * rmajor**2.11e0
        * kappa95**0.75e0
        * aspect ** (-0.3e0)
        * afuel**0.14e0
    )
