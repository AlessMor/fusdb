"""Spherical tokamak confinement scaling relations."""

from fusdb import relation


@relation(
    name="petty08_confinement_time",
    tags=("confinement", "spherical_tokamak", "h_mode"),
    outputs="tau_E",
)
def petty08_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
) -> float:
    """Calculate the beta independent dimensionless Petty08 confinement time
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
    kappa_ipb :
        IPB specific plasma separatrix elongation
    aspect :
        Aspect ratio

    Returns
    -------
    :
        float: Petty08 confinement time [s]

    Notes
    -----
        - This scaling uses the IPB defintiion of elongation, see reference for
          more information.

    References
    ----------
        - C. C. Petty, “Sizing up plasmas using dimensionless parameters,”
          Physics of Plasmas, vol. 15, no. 8, Aug. 2008,
          doi: https://doi.org/10.1063/1.2961043.

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
        0.052e0
        * pcur**0.75e0
        * b_plasma_toroidal_on_axis**0.3e0
        * dnla19**0.32e0
        * p_plasma_loss_mw ** (-0.47e0)
        * rmajor**2.09e0
        * kappa_ipb**0.88e0
        * aspect ** (-0.84e0)
    )


@relation(
    name="menard_nstx_confinement_time",
    tags=("confinement", "spherical_tokamak", "h_mode"),
    outputs="tau_E",
)
def menard_nstx_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the Menard NSTX ELMy H-mode scaling confinement time
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
    kappa_ipb :
        IPB specific plasma separatrix elongation
    aspect :
        Aspect ratio
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: Menard NSTX ELMy H-mode confinement time [s]

    Notes
    -----
        - "The leading NSTX conﬁnement scaling coefﬁcient is chosen such that the
           ITER and ST energy conﬁnement times are identical for a reference NSTX
           scenario"

        - Assumes IPB98(y,2) exponents are applicable where the ST exponents are
          not yet determined, i.e. the species mass, major radius, inverse aspect
          ratio and elongation. Hence here we use the IPB98(y,2) definition of
          elongation.

    References
    ----------
        - J. E. Menard, “Compact steady-state tokamak performance dependence on
          magnet and core physics limits,”
          Philosophical Transactions of the Royal Society A, vol. 377, no. 2141,
          pp. 20170440-20170440, Feb. 2019,
          doi: https://doi.org/10.1098/rsta.2017.0440.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.095e0
        * pcur**0.57e0
        * b_plasma_toroidal_on_axis**1.08e0
        * dnla19**0.44e0
        * p_plasma_loss_mw ** (-0.73e0)
        * rmajor**1.97e0
        * kappa_ipb**0.78e0
        * aspect ** (-0.58e0)
        * afuel**0.19e0
    )


@relation(
    name="menard_nstx_petty08_hybrid_confinement_time",
    tags=("confinement", "spherical_tokamak", "h_mode"),
    outputs="tau_E",
)
def menard_nstx_petty08_hybrid_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the Menard NSTX-Petty hybrid confinement time
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
    kappa_ipb :
        IPB specific plasma separatrix elongation
    aspect :
        Aspect ratio
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: Menard NSTX-Petty hybrid confinement time [s]

    Notes
    -----
        - Assuming a linear interpolation in (1/aspect) between the two scalings

    References
    ----------
        - J. E. Menard, “Compact steady-state tokamak performance dependence on
          magnet and core physics limits,”
          Philosophical Transactions of the Royal Society A, vol. 377, no. 2141,
          pp. 20170440-20170440, Feb. 2019,
          doi: https://doi.org/10.1098/rsta.2017.0440.
    """
    # Equivalent to A > 2.5, use Petty scaling
    if (1.0e0 / aspect) <= 0.4e0:
        return petty08_confinement_time.func(
            I_p,
            b_plasma_toroidal_on_axis,
            n_la,
            p_plasma_loss,
            rmajor,
            kappa_ipb,
            aspect,
        )

    #  Equivalent to A < 1.7, use NSTX scaling
    if (1.0e0 / aspect) >= 0.6e0:
        return menard_nstx_confinement_time.func(
            I_p,
            b_plasma_toroidal_on_axis,
            n_la,
            p_plasma_loss,
            rmajor,
            kappa_ipb,
            aspect,
            afuel,
        )
    return (((1.0e0 / aspect) - 0.4e0) / (0.6e0 - 0.4e0)) * (
        menard_nstx_confinement_time.func(
            I_p,
            b_plasma_toroidal_on_axis,
            n_la,
            p_plasma_loss,
            rmajor,
            kappa_ipb,
            aspect,
            afuel,
        )
    ) + ((0.6e0 - (1.0e0 / aspect)) / (0.6e0 - 0.4e0)) * (
        petty08_confinement_time.func(
            I_p,
            b_plasma_toroidal_on_axis,
            n_la,
            p_plasma_loss,
            rmajor,
            kappa_ipb,
            aspect,
        )
    )


@relation(
    name="nstx_gyro_bohm_confinement_time",
    tags=("confinement", "spherical_tokamak", "h_mode"),
    outputs="tau_E",
)
def nstx_gyro_bohm_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    rmajor: float,
    n_la: float,
) -> float:
    """Calculate the NSTX gyro-Bohm confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    rmajor :
        Plasma major radius [m]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3

    Returns
    -------
    :
        float: NSTX gyro-Bohm confinement time [s]

    References
    ----------
        - P. F. Buxton, L. Connor, A. E. Costley, Mikhail Gryaznevich, and
          S. McNamara, “On the energy confinement time in spherical tokamaks:
          implications for the design of pilot plants and fusion reactors,”
          vol. 61, no. 3, pp. 035006-035006, Jan. 2019,
          doi: https://doi.org/10.1088/1361-6587/aaf7e5.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.21e0
        * pcur**0.54e0
        * b_plasma_toroidal_on_axis**0.91e0
        * p_plasma_loss_mw ** (-0.38e0)
        * rmajor**2.14e0
        * dnla20 ** (-0.05e0)
    )
