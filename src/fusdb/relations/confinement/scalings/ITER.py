"""ITER-family confinement scaling relations."""

import numpy as np

from fusdb import relation


@relation(
    name="iter_89p_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E",
)
def iter_89p_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the ITER Power scaling - ITER 89-P (L-mode) confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa :
        Plasma elongation
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
        float: ITER 89-P confinement time [s]


    References
    ----------
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study',
          AEA FUS 172, 1992

        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.048e0
        * pcur**0.85e0
        * rmajor**1.2e0
        * rminor**0.3e0
        * np.sqrt(kappa)
        * dnla20**0.1e0
        * b_plasma_toroidal_on_axis**0.2e0
        * np.sqrt(afuel)
        / np.sqrt(p_plasma_loss_mw)
    )


@relation(
    name="iter_89_0_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E",
)
def iter_89_0_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the ITER Offset linear scaling - ITER 89-O (L-mode) confinement
      time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa :
        Plasma elongation
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
        float: ITER 89-O confinement time [s]

    References
    ----------
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study',
          AEA FUS 172, 1992
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    term1 = (
        0.04e0
        * pcur**0.5e0
        * rmajor**0.3e0
        * rminor**0.8e0
        * kappa**0.6e0
        * afuel**0.5e0
    )
    term2 = (
        0.064e0
        * pcur**0.8e0
        * rmajor**1.6e0
        * rminor**0.6e0
        * kappa**0.5e0
        * dnla20**0.6e0
        * b_plasma_toroidal_on_axis**0.35e0
        * afuel**0.2e0
        / p_plasma_loss_mw
    )
    return term1 + term2


@relation(
    name="iter_h90_p_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_h90_p_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the ITER H-mode scaling - ITER H90-P confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa :
        Plasma elongation
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
        float: ITER H90-P confinement time [s]


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
        0.064e0
        * pcur**0.87e0
        * rmajor**1.82e0
        * rminor ** (-0.12e0)
        * kappa**0.35e0
        * dnla20**0.09e0
        * b_plasma_toroidal_on_axis**0.15e0
        * np.sqrt(afuel)
        / np.sqrt(p_plasma_loss_mw)
    )


@relation(
    name="minimum_of_iter_89p_and_iter_89_0_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E",
)
def minimum_of_iter_89p_and_iter_89_0_confinement_time(
    I_p: float,
    rmajor: float,
    rminor: float,
    kappa: float,
    n_la: float,
    b_plasma_toroidal_on_axis: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the minimum of the ITER 89-P and ITER 89-O scalings.
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    rmajor :
        Plasma major radius [m]
    rminor :
        Plasma minor radius [m]
    kappa :
        Plasma elongation
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
        float: Minimum ITER 89-P and ITER 89-O confinement time [s]
    """
    return min(
        iter_89p_confinement_time.func(
            I_p,
            rmajor,
            rminor,
            kappa,
            n_la,
            b_plasma_toroidal_on_axis,
            afuel,
            p_plasma_loss,
        ),
        iter_89_0_confinement_time.func(
            I_p,
            rmajor,
            rminor,
            kappa,
            n_la,
            b_plasma_toroidal_on_axis,
            afuel,
            p_plasma_loss,
        ),
    )


@relation(
    name="iter_h90_p_amended_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_h90_p_amended_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    afuel: float,
    rmajor: float,
    p_plasma_loss: float,
    kappa: float,
) -> float:
    """Calculate the amended ITER H90-P confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    afuel :
        Fuel atomic mass number
    rmajor :
        Plasma major radius [m]
    p_plasma_loss_mw :
        Net Heating power [MW]
    kappa :
        Plasma elongation

    Returns
    -------
    :
        float: Amended ITER H90-P confinement time [s]

    References
    ----------
        - J. P. Christiansen et al., “Global energy confinement H-mode database
          for ITER,” Nuclear Fusion, vol. 32, no. 2, pp. 291-338, Feb. 1992,
          doi: https://doi.org/10.1088/0029-5515/32/2/i11.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    return (
        0.082e0
        * pcur**1.02e0
        * b_plasma_toroidal_on_axis**0.15e0
        * np.sqrt(afuel)
        * rmajor**1.60e0
        / (p_plasma_loss_mw**0.47e0 * kappa**0.19e0)
    )


@relation(
    name="iter_93h_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_93h_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    afuel: float,
    rmajor: float,
    n_la: float,
    aspect: float,
    kappa: float,
) -> float:
    """Calculate the ITER-93H scaling ELM-free confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    afuel :
        Fuel atomic mass number
    rmajor :
        Plasma major radius [m]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    aspect :
        Aspect ratio
    kappa :
        Plasma elongation

    Returns
    -------
    :
        float: ITER-93H confinement time [s]

    References
    ----------
        - K. Thomsen et al., “ITER H mode confinement database update,”
          vol. 34, no. 1, pp. 131-167, Jan. 1994,
          doi: https://doi.org/10.1088/0029-5515/34/1/i10.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.036e0
        * pcur**1.06e0
        * b_plasma_toroidal_on_axis**0.32e0
        * p_plasma_loss_mw ** (-0.67e0)
        * afuel**0.41e0
        * rmajor**1.79e0
        * dnla20**0.17e0
        * aspect**0.11e0
        * kappa**0.66e0
    )


@relation(
    name="iter_h97p_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_h97p_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    n_la: float,
    rmajor: float,
    aspect: float,
    kappa: float,
    afuel: float,
) -> float:
    """Calculate the ELM-free ITER H-mode scaling - ITER H97-P confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    rmajor :
        Plasma major radius [m]
    aspect :
        Aspect ratio
    kappa :
        Plasma elongation
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: ITER H97-P confinement time [s]

    References
    ----------
        - I. C. Database and M. W. G. (presented Cordey), “Energy confinement
          scaling and the extrapolation to ITER,”
          Plasma Physics and Controlled Fusion, vol. 39, no. 12B, pp. B115-B127,
          Dec. 1997, doi: https://doi.org/10.1088/0741-3335/39/12b/009.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.031e0
        * pcur**0.95e0
        * b_plasma_toroidal_on_axis**0.25e0
        * p_plasma_loss_mw ** (-0.67e0)
        * dnla19**0.35e0
        * rmajor**1.92e0
        * aspect ** (-0.08e0)
        * kappa**0.63e0
        * afuel**0.42e0
    )


@relation(
    name="iter_h97p_elmy_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_h97p_elmy_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    n_la: float,
    rmajor: float,
    aspect: float,
    kappa: float,
    afuel: float,
) -> float:
    """Calculate the ELMy ITER H-mode scaling - ITER H97-P(y) confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Net Heating power [MW]
    dnla19 :
        Line averaged electron density in units of 10**19 m**-3
    rmajor :
        Plasma major radius [m]
    aspect :
        Aspect ratio
    kappa :
        Plasma elongation
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: ITER H97-P(y) confinement time [s]

    References
    ----------
        - I. C. Database and M. W. G. (presented Cordey), “Energy confinement
          scaling and the extrapolation to ITER,”
          Plasma Physics and Controlled Fusion, vol. 39, no. 12B, pp. B115-B127,
          Dec. 1997, doi: https://doi.org/10.1088/0741-3335/39/12b/009.

        - International Atomic Energy Agency, Vienna (Austria), "Technical basis
          for the ITER final design report, cost review and safety analysis (FDR)",
          no.16. Dec. 1998.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.029e0
        * pcur**0.90e0
        * b_plasma_toroidal_on_axis**0.20e0
        * p_plasma_loss_mw ** (-0.66e0)
        * dnla19**0.40e0
        * rmajor**2.03e0
        * aspect ** (-0.19e0)
        * kappa**0.92e0
        * afuel**0.2e0
    )


@relation(
    name="iter_96p_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E",
)
def iter_96p_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    kappa95: float,
    rmajor: float,
    aspect: float,
    n_la: float,
    afuel: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the ITER-96P (= ITER-97L) L-mode scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    kappa95 :
        Plasma elongation at 95% flux surface
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
        float: ITER-96P confinement time [s]

    Notes
    -----
        - The thermal energy confinement time is given below

    References
    ----------
        - S. B. Kaye et al., “ITER L mode confinement database,”
        Nuclear Fusion, vol. 37, no. 9, pp. 1303-1328, Sep. 1997,
        doi: https://doi.org/10.1088/0029-5515/37/9/i10.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.023e0
        * pcur**0.96e0
        * b_plasma_toroidal_on_axis**0.03e0
        * kappa95**0.64e0
        * rmajor**1.83e0
        * aspect**0.06e0
        * dnla19**0.40e0
        * afuel**0.20e0
        * p_plasma_loss_mw ** (-0.73e0)
    )


@relation(
    name="iter_pb98py_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_pb98py_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the ITERH-PB98P(y) ELMy H-mode scaling confinement time
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
    kappa :
        Plasma separatrix elongation
    aspect :
        Aspect ratio
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: ITERH-PB98P(y) ELMy H-mode confinement time [s]
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0615e0
        * pcur**0.9e0
        * b_plasma_toroidal_on_axis**0.1e0
        * dnla19**0.4e0
        * p_plasma_loss_mw ** (-0.66e0)
        * rmajor**2
        * kappa**0.75e0
        * aspect ** (-0.66e0)
        * afuel**0.2e0
    )


@relation(
    name="iter_ipb98y_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_ipb98y_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the IPB98(y) ELMy H-mode scaling confinement time
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
    kappa :
        IPB sprcific plasma separatrix elongation
    aspect :
        Aspect ratio
    afuel :
        Fuel atomic mass number

    Returns
    -------
    :
        float: IPB98(y) ELMy H-mode confinement time [s]

    Notes
    -----
        - Unlike the other IPB98 scaling laws, the IPB98(y) scaling law uses the
          true separatrix elongation.
        - See correction paper below for more information

    References
    ----------
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B.
          Editors, “Chapter 2: Plasma confinement and transport,” Nuclear Fusion,
          vol. 39, no. 12, pp. 2175-2249, Dec. 1999,
          doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy, “Corrections to a
          sequence of papers in Nuclear Fusion,” Nuclear Fusion, vol. 48, no. 9,
          pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0365e0
        * pcur**0.97e0
        * b_plasma_toroidal_on_axis**0.08e0
        * dnla19**0.41e0
        * p_plasma_loss_mw ** (-0.63e0)
        * rmajor**1.93e0
        * kappa**0.67e0
        * aspect ** (-0.23e0)
        * afuel**0.2e0
    )


@relation(
    name="iter_ipb98y1_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_ipb98y1_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the IPB98(y,1) ELMy H-mode scaling confinement time
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
        float: IPB98(y,1) ELMy H-mode confinement time [s]

    Notes
    -----
        - See correction paper below for more information about the re-definition
          of the elongation used.

    References
    ----------
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B.
          Editors, “Chapter 2: Plasma confinement and transport,” Nuclear Fusion,
          vol. 39, no. 12, pp. 2175-2249, Dec. 1999,
          doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy, “Corrections to a
          sequence of papers in Nuclear Fusion,” Nuclear Fusion, vol. 48, no. 9,
          pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0503e0
        * pcur**0.91e0
        * b_plasma_toroidal_on_axis**0.15e0
        * dnla19**0.44e0
        * p_plasma_loss_mw ** (-0.65e0)
        * rmajor**2.05e0
        * kappa_ipb**0.72e0
        * aspect ** (-0.57e0)
        * afuel**0.13e0
    )


@relation(
    name="tau_E_iter_ipb98y2",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def tau_E_iter_ipb98y2(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the IPB98(y,2) ELMy H-mode scaling confinement time
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
        float: IPB98(y,2) ELMy H-mode confinement time [s]

    Notes
    -----
        - See correction paper below for more information about the re-definition
          of the elongation used.

    References
    ----------
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B.
          Editors, “Chapter 2: Plasma confinement and transport,” Nuclear Fusion,
          vol. 39, no. 12, pp. 2175-2249, Dec. 1999,
          doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy, “Corrections to a
          sequence of papers in Nuclear Fusion,” Nuclear Fusion, vol. 48, no. 9,
          pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0562e0
        * pcur**0.93e0
        * b_plasma_toroidal_on_axis**0.15e0
        * dnla19**0.41e0
        * p_plasma_loss_mw ** (-0.69e0)
        * rmajor**1.97e0
        * kappa_ipb**0.78e0
        * aspect ** (-0.58e0)
        * afuel**0.19e0
    )


@relation(
    name="iter_ipb98y2_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_ipb98y2_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the IPB98(y,2) ELMy H-mode scaling confinement time
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
        float: IPB98(y,2) ELMy H-mode confinement time [s]

    Notes
    -----
        - See correction paper below for more information about the re-definition
          of the elongation used.

    References
    ----------
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B.
          Editors, “Chapter 2: Plasma confinement and transport,” Nuclear Fusion,
          vol. 39, no. 12, pp. 2175-2249, Dec. 1999,
          doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy, “Corrections to a
          sequence of papers in Nuclear Fusion,” Nuclear Fusion, vol. 48, no. 9,
          pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0562e0
        * pcur**0.93e0
        * b_plasma_toroidal_on_axis**0.15e0
        * dnla19**0.41e0
        * p_plasma_loss_mw ** (-0.69e0)
        * rmajor**1.97e0
        * kappa_ipb**0.78e0
        * aspect ** (-0.58e0)
        * afuel**0.19e0
    )


@relation(
    name="iter_ipb98y3_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_ipb98y3_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the IPB98(y,3) ELMy H-mode scaling confinement time
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
        float: IPB98(y,3) ELMy H-mode confinement time [s]

    Notes
    -----
        - See correction paper below for more information about the re-definition
          of the elongation used.

    References
    ----------
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B.
          Editors, “Chapter 2: Plasma confinement and transport,” Nuclear Fusion,
          vol. 39, no. 12, pp. 2175-2249, Dec. 1999,
          doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy, “Corrections to a
          sequence of papers in Nuclear Fusion,” Nuclear Fusion, vol. 48, no. 9,
          pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0564e0
        * pcur**0.88e0
        * b_plasma_toroidal_on_axis**0.07e0
        * dnla19**0.40e0
        * p_plasma_loss_mw ** (-0.69e0)
        * rmajor**2.15e0
        * kappa_ipb**0.78e0
        * aspect ** (-0.64e0)
        * afuel**0.20e0
    )


@relation(
    name="iter_ipb98y4_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def iter_ipb98y4_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    kappa_ipb: float,
    aspect: float,
    afuel: float,
) -> float:
    """Calculate the IPB98(y,4) ELMy H-mode scaling confinement time
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
        float: IPB98(y,4) ELMy H-mode confinement time [s]

    Notes
    -----
        - See correction paper below for more information about the re-definition
          of the elongation used.

    References
    ----------
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B.
          Editors, “Chapter 2: Plasma confinement and transport,” Nuclear Fusion,
          vol. 39, no. 12, pp. 2175-2249, Dec. 1999,
          doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - Otto Kardaun, N. K. Thomsen, and Alexander Chudnovskiy, “Corrections to a
          sequence of papers in Nuclear Fusion,” Nuclear Fusion, vol. 48, no. 9,
          pp. 099801-099801, Aug. 2008,
          doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0587e0
        * pcur**0.85e0
        * b_plasma_toroidal_on_axis**0.29e0
        * dnla19**0.39e0
        * p_plasma_loss_mw ** (-0.70e0)
        * rmajor**2.08e0
        * kappa_ipb**0.76e0
        * aspect ** (-0.69e0)
        * afuel**0.17e0
    )


@relation(
    name="itpa20_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def itpa20_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
    rmajor: float,
    triang: float,
    kappa_ipb: float,
    eps: float,
    aion: float,
) -> float:
    """Calculate the ITPA20 Issue #3164 confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla19 :
        Central line-averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw :
        Thermal power lost due to transport through the LCFS [MW]
    rmajor :
        Plasma major radius [m]
    triang :
        Triangularity
    kappa_ipb :
        IPB specific plasma separatrix elongation
    eps :
        Inverse aspect ratio
    aion :
        Average mass of all ions (amu)

    Returns
    -------
    :
        float: ITPA20 confinement time [s]

    Notes
    -----
        - Mass term is the effective mass of the plasma, so we assume the total
          ion mass here
        - This scaling uses the IPB defintiion of elongation, see reference for
          more information.

    References
    ----------
        - G. Verdoolaege et al., “The updated ITPA global H-mode confinement
          database: description and analysis,” Nuclear Fusion, vol. 61, no. 7,
          pp. 076006-076006, Jan. 2021,
          doi: https://doi.org/10.1088/1741-4326/abdb91.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0534
        * pcur**0.976
        * b_plasma_toroidal_on_axis**0.218
        * dnla19**0.2442
        * p_plasma_loss_mw ** (-0.6687)
        * rmajor**1.710
        * (1 + triang) ** 0.362
        * kappa_ipb**0.799
        * eps**0.354
        * aion**0.195
    )


@relation(
    name="itpa20_il_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def itpa20_il_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    p_plasma_loss: float,
    n_la: float,
    aion: float,
    rmajor: float,
    triang: float,
    kappa_ipb: float,
) -> float:
    """Calculate the ITPA20-IL Issue #1852 confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    p_plasma_loss_mw :
        Thermal power lost due to transport through the LCFS [MW]
    dnla19 :
        Central line-averaged electron density in units of 10**19 m**-3
    aion :
        Average mass of all ions (amu)
    rmajor :
        Plasma major radius [m]
    triang :
        Triangularity
    kappa_ipb :
        IPB specific plasma separatrix elongation

    Returns
    -------
    :
        float: ITPA20-IL confinement time [s]

    Notes
    -----
        - Mass term is the effective mass of the plasma, so we assume the total
          ion mass here
        - This scaling uses the IPB defintiion of elongation, see reference for
          more information.

    References
    ----------
        - T. Luda et al., “Validation of a full-plasma integrated modeling approach
          on ASDEX Upgrade,” Nuclear Fusion, vol. 61, no. 12, pp. 126048-126048,
          Nov. 2021, doi: https://doi.org/10.1088/1741-4326/ac3293.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla19 = n_la / 1.0e19
    return (
        0.0670
        * pcur**1.291
        * b_plasma_toroidal_on_axis**-0.134
        * dnla19**0.1473
        * p_plasma_loss_mw ** (-0.6442)
        * rmajor**1.194
        * (1 + triang) ** 0.560
        * kappa_ipb**0.673
        * aion**0.302
    )


@relation(
    name="cfspopcon_iter89p_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E",
)
def cfspopcon_iter89p_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa_ipb: float,
    n_avg: float,
) -> float:
    """Calculate the ITER89P confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - Yushmanov et al, 'Scalings for tokamak energy confinement'
          Nuclear Fusion, vol. 30, no. 10, pp. 4-6, 1990.

    Notes
    -----
        - C is corrected for average_electron_density convention. N.b. The
          different factor of a_R is because we use inverse_aspect_ratio=a/R
          instead of a. R^1.2 a^0.3 = R^1.5 inverse_aspect_ratio^0.3.
        - Regime: L-Mode
    """
    return (
        H98_y2
        * 0.03812775526676551
        * afuel**0.5
        * B0**0.2
        * (I_p / 1.0e6) ** 0.85
        * (P_loss / 1.0e6) ** (-0.5)
        * R**1.5
        * eps**0.3
        * kappa_ipb**0.5
        * (n_avg / 1.0e19) ** 0.1
    )


@relation(
    name="cfspopcon_iter89p_ka_confinement_time",
    tags=("confinement", "l_mode"),
    outputs="tau_E",
)
def cfspopcon_iter89p_ka_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITER89P_ka confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - Yushmanov et al, 'Scalings for tokamak energy confinement'
          Nuclear Fusion, vol. 30, no. 10, pp. 4-6, 1990.

    Notes
    -----
        - C is corrected for average_electron_density convention. Using kappa_A
          instead of separatrix_elongation, which is likely more accurate for
          double-nulls. N.b. The different factor of a_R is because we use
          inverse_aspect_ratio=a/R instead of a. R^1.2 a^0.3 = R^1.5
          inverse_aspect_ratio^0.3.
        - Regime: L-Mode
    """
    return (
        H98_y2
        * 0.03812775526676551
        * afuel**0.5
        * B0**0.2
        * (I_p / 1.0e6) ** 0.85
        * (P_loss / 1.0e6) ** (-0.5)
        * R**1.5
        * eps**0.3
        * kappa**0.5
        * (n_avg / 1.0e19) ** 0.1
    )


@relation(
    name="cfspopcon_itpa_2018_std5_gls_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa_2018_std5_gls_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA_2018_STD5_GLS confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITPA 2018 STD5-GLS Verdoolaege et al, 'First Analysis of the Updated
          ITPA Global H-mode Confinement Database', 2018, International Atomic
          Energy Agency.

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.042
        * afuel**0.47
        * B0**0.068
        * (I_p / 1.0e6) ** 1.2
        * (P_loss / 1.0e6) ** (-0.78)
        * R**1.6
        * eps ** (-0.052)
        * kappa**0.88
        * (n_avg / 1.0e19) ** 0.21
    )


@relation(
    name="cfspopcon_itpa_2018_std5_ols_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa_2018_std5_ols_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA_2018_STD5_OLS confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITPA 2018 STD5-OLS Verdoolaege et al, 'First Analysis of the Updated
          ITPA Global H-mode Confinement Database', 2018, International Atomic
          Energy Agency.

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.049
        * afuel**0.25
        * B0**0.085
        * (I_p / 1.0e6) ** 1.1
        * (P_loss / 1.0e6) ** (-0.71)
        * R**1.5
        * eps ** (-0.043)
        * kappa**0.8
        * (n_avg / 1.0e19) ** 0.19
    )


@relation(
    name="cfspopcon_itpa_2018_std5_sel1_gls_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa_2018_std5_sel1_gls_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA_2018_STD5_SEL1_GLS confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITPA 2018 STD5-SEL1-GLS Verdoolaege et al, 'First Analysis of the
          Updated ITPA Global H-mode Confinement Database', 2018,
          International Atomic Energy Agency.

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.023
        * afuel**0.33
        * B0 ** (-0.018)
        * (I_p / 1.0e6) ** 1.3
        * (P_loss / 1.0e6) ** (-0.79)
        * R**1.5
        * eps ** (-0.38)
        * kappa**1.9
        * (n_avg / 1.0e19) ** 0.17
    )


@relation(
    name="cfspopcon_itpa_2018_std5_sel1_ols_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa_2018_std5_sel1_ols_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA_2018_STD5_SEL1_OLS confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITPA 2018 STD5-SEL1-OLS Verdoolaege et al, 'First Analysis of the
          Updated ITPA Global H-mode Confinement Database', 2018,
          International Atomic Energy Agency.

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.045
        * afuel**0.24
        * B0 ** (-0.1)
        * (I_p / 1.0e6) ** 1.3
        * (P_loss / 1.0e6) ** (-0.71)
        * R**1.2
        * eps ** (-0.32)
        * kappa**1.1
        * (n_avg / 1.0e19) ** 0.13
    )


@relation(
    name="cfspopcon_itpa_2018_std5_sel1_wls_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa_2018_std5_sel1_wls_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA_2018_STD5_SEL1_WLS confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITPA 2018 STD5-SEL1-WLS Verdoolaege et al, 'First Analysis of the
          Updated ITPA Global H-mode Confinement Database', 2018,
          International Atomic Energy Agency.

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.03
        * afuel**0.094
        * B0 ** (-0.069)
        * (I_p / 1.0e6) ** 1.3
        * (P_loss / 1.0e6) ** (-0.64)
        * R**1.3
        * eps ** (-0.46)
        * kappa**1.3
        * (n_avg / 1.0e19) ** 0.19
    )


@relation(
    name="cfspopcon_itpa_2018_std5_wls_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa_2018_std5_wls_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA_2018_STD5_WLS confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITPA 2018 STD5-WLS Verdoolaege et al, 'First Analysis of the Updated
          ITPA Global H-mode Confinement Database', 2018, International Atomic
          Energy Agency.

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.04
        * afuel**0.25
        * B0**0.11
        * (I_p / 1.0e6) ** 0.99
        * (P_loss / 1.0e6) ** (-0.64)
        * R**1.7
        * eps**0.093
        * kappa**0.79
        * (n_avg / 1.0e19) ** 0.29
    )


@relation(
    name="cfspopcon_itpa20_il_highz_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa20_il_highz_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA20_IL_HighZ confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITER H20, DB5.2.3, High Z walls only, G. Verdoolaege et al 2021
          Nucl. Fusion 61 076006, 'The updated ITPA global H-mode confinement
          database: description and analysis'

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.189
        * afuel**0.312
        * B0 ** (-0.356)
        * (I_p / 1.0e6) ** 1.485
        * (P_loss / 1.0e6) ** (-0.6077)
        * R**0.671
        * (n_avg / 1.0e19) ** 0.018
    )


@relation(
    name="cfspopcon_itpa20_il_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa20_il_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    delta_95: float,
    delta: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA20_IL confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITER H20, DB5.2.3, ITER-like discharges, G. Verdoolaege et al 2021
          Nucl. Fusion 61 076006, 'The updated ITPA global H-mode confinement
          database: description and analysis'

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.067
        * afuel**0.3
        * B0 ** (-0.13)
        * (I_p / 1.0e6) ** 1.29
        * (P_loss / 1.0e6) ** (-0.644)
        * R**1.19
        * (1.0 + np.mean([delta_95, delta])) ** 0.56
        * kappa**0.67
        * (n_avg / 1.0e19) ** 0.15
    )


@relation(
    name="cfspopcon_itpa20_std5_confinement_time",
    tags=("confinement", "h_mode"),
    outputs="tau_E",
)
def cfspopcon_itpa20_std5_confinement_time(
    H98_y2: float,
    afuel: float,
    B0: float,
    I_p: float,
    P_loss: float,
    R: float,
    delta_95: float,
    delta: float,
    eps: float,
    kappa: float,
    n_avg: float,
) -> float:
    """Calculate the ITPA20_STD5 confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - ITER H20, DB5.2.3, STD5 discharges, G. Verdoolaege et al 2021
          Nucl. Fusion 61 076006, 'The updated ITPA global H-mode confinement
          database: description and analysis'

    Notes
    -----
        - Regime: H-Mode
    """
    return (
        H98_y2
        * 0.053
        * afuel**0.2
        * B0**0.22
        * (I_p / 1.0e6) ** 0.98
        * (P_loss / 1.0e6) ** (-0.669)
        * R**1.71
        * (1.0 + np.mean([delta_95, delta])) ** 0.36
        * eps**0.35
        * kappa**0.80
        * (n_avg / 1.0e19) ** 0.24
    )
