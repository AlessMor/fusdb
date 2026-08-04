"""I-mode confinement scaling relations."""

from fusdb.relation import relation

@relation(
    name="hubbard_nominal_confinement_time",
    tags=("confinement", "i_mode"),
    outputs="tau_E",
    h_factor="H_hubbard_nominal",
)
def hubbard_nominal_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Hubbard 2017 I-mode confinement time scaling - nominal
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Hubbard confinement time [s]

    References
    ----------
        - A. E. Hubbard et al., “Physics and performance of the I-mode regime over
          an expanded operating space on Alcator C-Mod,” Nuclear Fusion, vol. 57,
          no. 12, p. 126039, Oct. 2017,
          doi: https://doi.org/10.1088/1741-4326/aa8570.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.014e0
        * pcur**0.68e0
        * b_plasma_toroidal_on_axis**0.77e0
        * dnla20**0.02e0
        * p_plasma_loss_mw ** (-0.29e0)
    )


@relation(
    name="hubbard_lower_confinement_time",
    tags=("confinement", "i_mode"),
    outputs="tau_E",
    h_factor="H_hubbard_lower",
)
def hubbard_lower_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Hubbard 2017 I-mode confinement time scaling - lower
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Hubbard confinement time [s]

    References
    ----------
        - A. E. Hubbard et al., “Physics and performance of the I-mode regime over
          an expanded operating space on Alcator C-Mod,” Nuclear Fusion, vol. 57,
          no. 12, p. 126039, Oct. 2017,
          doi: https://doi.org/10.1088/1741-4326/aa8570.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.014e0
        * pcur**0.60e0
        * b_plasma_toroidal_on_axis**0.70e0
        * dnla20 ** (-0.03e0)
        * p_plasma_loss_mw ** (-0.33e0)
    )

@relation(
    name="hubbard_upper_confinement_time",
    tags=("confinement", "i_mode"),
    outputs="tau_E",
    h_factor="H_hubbard_upper",
)
def hubbard_upper_confinement_time(
    I_p: float,
    b_plasma_toroidal_on_axis: float,
    n_la: float,
    p_plasma_loss: float,
) -> float:
    """Calculate the Hubbard 2017 I-mode confinement time scaling - upper
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    pcur :
        Plasma current [MA]
    b_plasma_toroidal_on_axis :
        Toroidal magnetic field [T]
    dnla20 :
        Line averaged electron density in units of 10**20 m**-3
    p_plasma_loss_mw :
        Net Heating power [MW]

    Returns
    -------
    :
        float: Hubbard confinement time [s]

    References
    ----------
        - A. E. Hubbard et al., “Physics and performance of the I-mode regime over
          an expanded operating space on Alcator C-Mod,” Nuclear Fusion, vol. 57,
          no. 12, p. 126039, Oct. 2017,
          doi: https://doi.org/10.1088/1741-4326/aa8570.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    pcur = I_p / 1.0e6
    p_plasma_loss_mw = p_plasma_loss / 1.0e6
    dnla20 = n_la / 1.0e20
    return (
        0.014e0
        * pcur**0.76e0
        * b_plasma_toroidal_on_axis**0.84e0
        * dnla20**0.07
        * p_plasma_loss_mw ** (-0.25e0)
    )


@relation(
    name="cfspopcon_imodey2_confinement_time",
    tags=("confinement", "i_mode", "confinement_mode_default"),
    outputs="tau_E",
)
def cfspopcon_imodey2_confinement_time(
    H98_y2: float,
    B0: float,
    I_p: float,
    P_loss: float,
    n_avg: float,
) -> float:
    """Calculate the IModey2 confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - Walk, J. R., Pedestal structure and stability in high-performance
          plasmas on Alcator C-Mod, https://dspace.mit.edu/handle/1721.1/95524,
          equation 5.2

    Notes
    -----
        - Coefficient C adjusted to account for ne in 1e19m^-3
        - Regime: I-Mode
    """
    return (
        H98_y2
        * 0.01346
        * B0**0.768
        * (I_p / 1.0e6) ** 0.685
        * (P_loss / 1.0e6) ** (-0.286)
        * (n_avg / 1.0e19) ** 0.017
    )
