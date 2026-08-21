"""L-mode confinement scaling relations."""

import numpy as np

from fusdb.relation import relation


@relation(
    name="merezhkin_muhkovatov_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def merezhkin_muhkovatov_confinement_time(
    rmajor: float,
    rminor: float,
    kappa95: float,
    qstar: float,
    n_la: float,
    afuel: float,
    ten: float,
) -> float:
    """Calculate the Merezhkin-Mukhovatov (MM) OH/L-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa95 :
        Plasma elongation at 95% flux surface
    qstar :
        Equivalent cylindrical edge safety factor
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    afuel :
        Fuel atomic mass number
    ten :
        Electron temperature [keV]

    Returns
    -------
    :
        float: Merezhkin-Mukhovatov confinement time [s]


    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    dnla20 = n_la / 1.0e20
    return (
        3.5e-3
        * rmajor**2.75e0
        * rminor**0.25e0
        * kappa95**0.125e0
        * qstar
        * dnla20
        * np.sqrt(afuel)
        / np.sqrt(ten / 10.0e0)
    )




@relation(
    name="rebut_lallia_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def rebut_lallia_confinement_time(
    rminor: float,
    rmajor: float,
    kappa: float,
    afuel: float,
    I_p: float,
    zeff: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Rebut-Lallia offset linear scaling (L-mode) confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    rminor :
        Plasma minor radius [m]
    rmajor :
        Plasma major radius [m]
    kappa :
        Plasma elongation at 95% flux surface
    afuel :
        Fuel atomic mass number
    pcur :
        Plasma current [MA]
    zeff :
        Effective charge
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Rebut-Lallia confinement time [s]


    References
    ----------
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    rll = (rminor**2 * rmajor * kappa) ** (1.0e0 / 3.0e0)
    term1 = 1.2e-2 * pcur * rll**1.5e0 / np.sqrt(zeff)
    term2 = (
        0.146e0
        * dnla20**0.75e0
        * np.sqrt(pcur)
        * np.sqrt(b_plasma_toroidal_on_axis)
        * rll**2.75e0
        * zeff**0.25e0
        / p_plasma_loss_mw
    )
    return 1.65e0 * np.sqrt(afuel / 2.0e0) * (term1 + term2)


@relation(
    name="goldston_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def goldston_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa95: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Goldston scaling (L-mode) confinement time
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
    afuel :
        Fuel atomic mass number
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Goldston confinement time [s]

    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    return (
        0.037e0
        * pcur
        * rmajor**1.75e0
        * rminor ** (-0.37e0)
        * np.sqrt(kappa95)
        * np.sqrt(afuel / 1.5e0)
        / np.sqrt(p_plasma_loss_mw)
    )


@relation(
    name="kaye_goldston_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def kaye_goldston_confinement_time(
    kappa95: float,
    I_p: float,
    n_la: float,
    rmajor: float,
    afuel: float,
    b_plasma_toroidal_on_axis: float,
    rminor: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Kaye-Goldston (KG) L-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    kappa95 :
        Plasma elongation at 95% flux surface
    pcur :
        Plasma current [MA]
    n20 :
        Line averaged electron density in units of 10**20 m**-3
    rmajor :
        Plasma major radius [m]
    afuel :
        Fuel atomic mass number
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    rminor :
        Plasma minor radius [m]
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Kaye-Goldston confinement time [s]

    Notes
    -----
        - An isotope correction factor (M_i/1.5)^0.5 is added to the original
          scaling to reflect the fact that the empirical fits to the data were
          from experiments with H and D  mixture, M_i = 1.5

    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    n20 = n_la / 1.0e20
    return (
        0.055e0
        * kappa95**0.28e0
        * pcur**1.24e0
        * n20**0.26e0
        * rmajor**1.65e0
        * np.sqrt(afuel / 1.5e0)
        / (
            b_plasma_toroidal_on_axis**0.09e0
            * rminor**0.49e0
            * p_plasma_loss_mw**0.58e0
        )
    )
    
    
@relation(
    name="t10_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def t10_confinement_time(
    n_la: float,
    rmajor: float,
    qstar: float,
    b_plasma_toroidal_on_axis: float,
    rminor: float,
    kappa95: float,
    p_plasma_loss: float,
    zeff: float,
    I_p: float,
) -> float:
    """Calculate the T-10 scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    rmajor :
        Plasma major radius [m]
    qstar :
        Equivalent cylindrical edge safety factor
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    rminor :
        Plasma minor radius [m]
    kappa95 :
        Plasma elongation at 95% flux surface
    p_plasma_loss_mw :
        Net Heating power [MW]
    zeff :
        Effective charge
    pcur :
        Plasma current [MA]

    Returns
    -------
    :
        float: T-10 confinement time [s]

    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    denfac = dnla20 * rmajor * qstar / (1.3e0 * b_plasma_toroidal_on_axis)
    denfac = min(1.0e0, denfac)
    return (
        0.095e0
        * rmajor
        * rminor
        * b_plasma_toroidal_on_axis
        * np.sqrt(kappa95)
        * denfac
        / p_plasma_loss_mw**0.4e0
        * (zeff**2 * pcur**4 / (rmajor * rminor * qstar**3 * kappa95**1.5e0))
        ** 0.08e0
    )


@relation(
    name="jaeri_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def jaeri_confinement_time(
    kappa95: float,
    rminor: float,
    afuel: float,
    n_la: float,
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    rmajor: float,
    qstar: float,
    zeff: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the JAERI / Odajima-Shimomura L-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    kappa95 :
        Plasma elongation at 95% flux surface
    rminor :
        Plasma minor radius [m]
    afuel :
        Fuel atomic mass number
    n20 :
        Line averaged electron density in units of 10**20 m**-3
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    rmajor :
        Plasma major radius [m]
    qstar :
        Equivalent cylindrical edge safety factor
    zeff :
        Effective charge
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: JAERI confinement time [s]

    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria) and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    n20 = n_la / 1.0e20
    gjaeri = (
        zeff**0.4e0
        * ((15.0e0 - zeff) / 20.0e0) ** 0.6e0
        * (3.0e0 * qstar * (qstar + 5.0e0) / ((qstar + 2.0e0) * (qstar + 7.0e0)))
        ** 0.6e0
    )
    return (
        0.085e0 * kappa95 * rminor**2 * np.sqrt(afuel)
        + 0.069e0
        * n20**0.6e0
        * pcur
        * b_plasma_toroidal_on_axis**0.2e0
        * rminor**0.4e0
        * rmajor**1.6e0
        * np.sqrt(afuel)
        * gjaeri
        * kappa95**0.2e0
        / p_plasma_loss_mw
    )


@relation(
    name="kaye_big_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def kaye_big_confinement_time(
    rmajor: float,
    rminor: float,
    b_plasma_toroidal_on_axis: float,
    kappa95: float,
    I_p: float,
    n_la: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Kaye-Big scaling confinement time
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
    pcur :
        Plasma current [MA]
    n20 :
        Line averaged electron density in units of 10**20 m**-3
    afuel :
        Fuel atomic mass number
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Kaye-Big confinement time [s]


    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria) and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    n20 = n_la / 1.0e20
    return (
        0.105e0
        * np.sqrt(rmajor)
        * rminor**0.8e0
        * b_plasma_toroidal_on_axis**0.3e0
        * kappa95**0.25e0
        * pcur**0.85e0
        * n20**0.1e0
        * np.sqrt(afuel)
        / np.sqrt(p_plasma_loss_mw)
    )


@relation(
    name="riedel_l_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def riedel_l_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa95: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Riedel scaling (L-mode) confinement time
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
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Riedel confinement time [s]

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
        0.044e0
        * pcur**0.93e0
        * rmajor**1.37e0
        * rminor ** (-0.049e0)
        * kappa95**0.588e0
        * dnla20**0.078e0
        * b_plasma_toroidal_on_axis**0.152e0
        / p_plasma_loss_mw**0.537e0
    )


@relation(
    name="christiansen_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def christiansen_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa95: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    afuel: float,
) -> float:
    """Calculate the Christiansen et al scaling (L-mode) confinement time
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
    p_plasma_loss_mw :
        Net Heating power [MW]
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: Christiansen confinement time [s]

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
        0.24e0
        * pcur**0.79e0
        * rmajor**0.56e0
        * rminor**1.46e0
        * kappa95**0.73e0
        * dnla20**0.41e0
        * b_plasma_toroidal_on_axis**0.29e0
        / (p_plasma_loss_mw**0.79e0 * afuel**0.02e0)
    )


@relation(
    name="lackner_gottardi_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def lackner_gottardi_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa95: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Lackner-Gottardi scaling (L-mode) confinement time
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
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Lackner-Gottardi confinement time [s]

    References
    ----------
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    qhat = (
        (1.0e0 + kappa95**2)
        * rminor**2
        * b_plasma_toroidal_on_axis
        / (0.4e0 * pcur * rmajor)
    )
    return (
        0.12e0
        * pcur**0.8e0
        * rmajor**1.8e0
        * rminor**0.4e0
        * kappa95
        * (1.0e0 + kappa95) ** (-0.8e0)
        * dnla20**0.6e0
        * qhat**0.4e0
        / p_plasma_loss_mw**0.6e0
    )


@relation(
    name="neo_kaye_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def neo_kaye_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa95: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Neo-Kaye scaling (L-mode) confinement time
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
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Neo-Kaye confinement time [s]


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
        0.063e0
        * pcur**1.12e0
        * rmajor**1.3e0
        * rminor ** (-0.04e0)
        * kappa95**0.28e0
        * dnla20**0.14e0
        * b_plasma_toroidal_on_axis**0.04e0
        / p_plasma_loss_mw**0.59e0
    )




@relation(
    name="kaye_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E_scaling",
)
def kaye_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    kappa: float,
    rmajor: float,
    aspect: float,
    n_la: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Kaye PPPL Workshop April 1998 L-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    kappa :
        Plasma elongation
    rmajor :
        Plasma major radius [m]
    aspect :
        Aspect ratio
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    afuel :
        Fuel atomic mass number
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Kaye PPPL Workshop confinement time [s]

    References
    ----------
        - Kaye PPPL Workshop April 1998
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.021e0
        * pcur**0.81e0
        * b_plasma_toroidal_on_axis**0.14e0
        * kappa**0.7e0
        * rmajor**2.01e0
        * aspect ** (-0.18e0)
        * dnla19**0.47e0
        * afuel**0.25e0
        * p_plasma_loss_mw ** (-0.73e0)
    )
