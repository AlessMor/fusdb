"""Energy confinement time scalings defined as independent relations.
Most of them are taken as-is from the PROCESS code (UKAEA)."""

from __future__ import annotations

import sympy as sp

from fusdb.relation_class import Relation_decorator as Relation
# Only the active default scaling is decorated for relation discovery.


############################################### TO BE IMPLEMENTED ###############################################

def tau_E_itpa20_il(
    I_p: float, B0: float, P_loss: float, n_la: float, aion: float, R: float, delta: float, kappa_ipb: float
) -> float:
    """
    Calculate the ITPA20-IL Issue #1852 confinement time

    Args:
        pcur (float): Plasma current [A]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        p_plasma_loss (float): Thermal power lost due to transport through the LCFS [W]
        n_la (float): Central line-averaged electron density in m**-3
        aion (float): Average mass of all ions (amu)
        rmajor (float): Plasma major radius [m]
        triang (float): Triangularity
        kappa_ipb (float): IPB specific plasma separatrix elongation

    Returns:
        float: ITPA20-IL confinement time [s]

    Notes:
        Mass term is the effective mass of the plasma, so we assume the total ion mass here
        This scaling uses the IPB defintiion of elongation, see reference for more information.

    References:
        T. Luda et al., “Validation of a full-plasma integrated modeling approach on ASDEX Upgrade,”
        Nuclear Fusion, vol. 61, no. 12, pp. 126048-126048, Nov. 2021, doi: https://doi.org/10.1088/1741-4326/ac3293.
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.067
        * I_p_MA ** 1.29
        * B0 ** -0.13
        * P_loss_MW ** (-0.644)
        * dnla19 ** 0.15
        * aion ** 0.3
        * R ** 1.19
        * (1 + delta) ** 0.56
        * kappa_ipb ** 0.67
    )



def tau_E_itpa20(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, delta: float, kappa_ipb: float, eps: float, aion: float
) -> float:
    """
    Calculate the ITPA20 Issue #3164 confinement time

    Args:
        pcur (float): Plasma current [A]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        n_la (float): Central line-averaged electron density in m**-3
        p_plasma_loss (float): Thermal power lost due to transport through the LCFS [W]
        rmajor (float): Plasma major radius [m]
        triang (float): Triangularity
        kappa_ipb (float): IPB specific plasma separatrix elongation
        eps (float): Inverse aspect ratio
        aion (float): Average mass of all ions (amu)

    Returns:
        float: ITPA20 confinement time [s]

    Notes:
        Mass term is the effective mass of the plasma, so we assume the total ion mass here
        This scaling uses the IPB defintiion of elongation, see reference for more information.

    References:
        G. Verdoolaege et al., “The updated ITPA global H-mode confinement database: description and analysis,”
        Nuclear Fusion, vol. 61, no. 7, pp. 076006-076006, Jan. 2021, doi: https://doi.org/10.1088/1741-4326/abdb91.
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.053
        * I_p_MA ** 0.98
        * B0 ** 0.22
        * dnla19 ** 0.24
        * P_loss_MW ** (-0.669)
        * R ** 1.71
        * (1 + delta) ** 0.36
        * kappa_ipb ** 0.8
        * eps ** 0.35
        * aion ** 0.2
    )




def tau_E_itpa20_il_high_z(I_p: float, B0: float, P_loss: float, n_la: float, aion: float, R: float) -> float:
    """ITPA20-IL scaling (high-Z subset), coefficients adjusted for density in 1e19 m^-3."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.189
        * I_p_MA ** 1.485
        * B0 ** -0.356
        * P_loss_MW ** (-0.6077)
        * dnla19 ** 0.018
        * aion ** 0.312
        * R ** 0.671
    )



def tau_E_loc(dnla20: float, q_star: float, kappa_A: float, eps: float, R: float) -> float:
    """
    Linear Ohmic Confinement scaling (LOC) from Rice et al. 2020.

    Uses density in 1e20 m^-3, inverse aspect ratio eps=a/R, and areal elongation kappa_A.
    """
    return 0.007 * dnla20 * q_star * kappa_A ** 0.5 * eps * R ** 3.0


def tau_E_nstx_gyro_bohm(I_p: float, B0: float, P_loss: float, R: float, dnla20: float) -> float:
    """
        Calculate the NSTX gyro-Bohm confinement time

        Args:
            pcur (float): Plasma current [A]
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            p_plasma_loss (float): Net Heating power [W]
            rmajor (float): Plasma major radius [m]
            dnla20 (float): Line averaged electron density in units of 10**20 m**-3

        Returns:
            float: NSTX gyro-Bohm confinement time [s]

        Notes:

        References:
            P. F. Buxton, L. Connor, A. E. Costley, Mikhail Gryaznevich, and S. McNamara,
            “On the energy confinement time in spherical tokamaks: implications for the design of pilot plants and fusion reactors,”
            vol. 61, no. 3, pp. 035006-035006, Jan. 2019, doi: https://doi.org/10.1088/1361-6587/aaf7e5.
            ‌
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return 0.21 * I_p_MA ** 0.54 * B0 ** 0.91 * P_loss_MW ** (-0.38) * R ** 2.14 * dnla20 ** (-0.05)




def tau_E_itpa_2018_std5_gls(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITPA 2018 STD5-GLS scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    eps_safe = eps
    return (
        0.042
        * afuel ** 0.47
        * B0 ** 0.068
        * I_p_MA ** 1.2
        * P_loss_MW ** (-0.78)
        * R ** 1.6
        * eps_safe ** -0.052
        * kappa_A ** 0.88
        * dnla19 ** 0.21
    )


def tau_E_itpa_2018_std5_ols(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITPA 2018 STD5-OLS scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    eps_safe = eps
    return (
        0.049
        * afuel ** 0.25
        * B0 ** 0.085
        * I_p_MA ** 1.1
        * P_loss_MW ** (-0.71)
        * R ** 1.5
        * eps_safe ** -0.043
        * kappa_A ** 0.8
        * dnla19 ** 0.19
    )


def tau_E_itpa_2018_std5_sel1_gls(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITPA 2018 STD5-SEL1-GLS scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    eps_safe = eps
    return (
        0.023
        * afuel ** 0.33
        * B0 ** -0.018
        * I_p_MA ** 1.3
        * P_loss_MW ** (-0.79)
        * R ** 1.5
        * eps_safe ** -0.38
        * kappa_A ** 1.9
        * dnla19 ** 0.17
    )


def tau_E_itpa_2018_std5_sel1_ols(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITPA 2018 STD5-SEL1-OLS scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    eps_safe = eps
    return (
        0.045
        * afuel ** 0.24
        * B0 ** -0.1
        * I_p_MA ** 1.3
        * P_loss_MW ** (-0.71)
        * R ** 1.2
        * eps_safe ** -0.32
        * kappa_A ** 1.1
        * dnla19 ** 0.13
    )


def tau_E_itpa_2018_std5_sel1_wls(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITPA 2018 STD5-SEL1-WLS scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    eps_safe = eps
    return (
        0.03
        * afuel ** 0.094
        * B0 ** -0.069
        * I_p_MA ** 1.3
        * P_loss_MW ** (-0.64)
        * R ** 1.3
        * eps_safe ** -0.46
        * kappa_A ** 1.3
        * dnla19 ** 0.19
    )


def tau_E_itpa_2018_std5_wls(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITPA 2018 STD5-WLS scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.04
        * afuel ** 0.25
        * B0 ** 0.11
        * I_p_MA ** 0.99
        * P_loss_MW ** (-0.64)
        * R ** 1.7
        * eps ** 0.093
        * kappa_A ** 0.79
        * dnla19 ** 0.29
    )


def tau_E_menard_nstx_petty08_hybrid(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """Return tau E menard nstx petty08 hybrid scaling."""
    dnla19 = n_la / 1e19
    invA = 1.0 / A
    if invA <= 0.4:
        return tau_E_petty08(I_p, B0, n_la, P_loss, R, kappa_ipb, A)
    if invA >= 0.6:
        return tau_E_menard_nstx(I_p, B0, n_la, P_loss, R, kappa_ipb, A, afuel)
    w = (invA - 0.4) / (0.6 - 0.4)
    return w * tau_E_menard_nstx(I_p, B0, n_la, P_loss, R, kappa_ipb, A, afuel) + (1 - w) * tau_E_petty08(I_p, B0, n_la, P_loss, R, kappa_ipb, A)




def tau_E_menard_nstx(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """Return tau E menard nstx scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.095
        * I_p_MA ** 0.57
        * B0 ** 1.08
        * dnla19 ** 0.44
        * P_loss_MW ** (-0.73)
        * R ** 1.97
        * kappa_ipb ** 0.78
        * A ** (-0.58)
        * afuel ** 0.19
    )




def tau_E_hubbard_upper(I_p: float, B0: float, dnla20: float, P_loss: float) -> float:
    """
        Calculate the Hubbard 2017 I-mode confinement time scaling - upper

        Args:
            pcur (float): Plasma current [A]
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            dnla20 (float): Line averaged electron density in units of 10**20 m**-3
            p_plasma_loss (float): Net Heating power [W]

        Returns:
            float: Hubbard confinement time [s]

        Notes:

        References:
            A. E. Hubbard et al., “Physics and performance of the I-mode regime over an expanded operating space on Alcator C-Mod,”
            Nuclear Fusion, vol. 57, no. 12, p. 126039, Oct. 2017, doi: https://doi.org/10.1088/1741-4326/aa8570.
            ‌
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return 0.014 * I_p_MA ** 0.76 * B0 ** 0.84 * dnla20 ** 0.07 * P_loss_MW ** (-0.25)




def tau_E_hubbard_lower(I_p: float, B0: float, dnla20: float, P_loss: float) -> float:
    """
        Calculate the Hubbard 2017 I-mode confinement time scaling - lower

        Args:
            pcur (float): Plasma current [A]
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            dnla20 (float): Line averaged electron density in units of 10**20 m**-3
            p_plasma_loss (float): Net Heating power [W]

        Returns:
            float: Hubbard confinement time [s]

        Notes:

        References:
            A. E. Hubbard et al., “Physics and performance of the I-mode regime over an expanded operating space on Alcator C-Mod,”
            Nuclear Fusion, vol. 57, no. 12, p. 126039, Oct. 2017, doi: https://doi.org/10.1088/1741-4326/aa8570.
            ‌
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return 0.014 * I_p_MA ** 0.60 * B0 ** 0.70 * dnla20 ** (-0.03) * P_loss_MW ** (-0.33)



@Relation(
    name="tau_E_hubbard_nominal",
    output="tau_E",
    tags=("confinement", "tokamak", "imode"),
)
def tau_E_hubbard_nominal(I_p: float, B0: float, dnla20: float, P_loss: float) -> float:
    """
        Calculate the Hubbard 2017 I-mode confinement time scaling - nominal

        Args:
            pcur (float): Plasma current [A]
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            dnla20 (float): Line averaged electron density in units of 10**20 m**-3
            p_plasma_loss (float): Net Heating power [W]

        Returns:
            float: Hubbard confinement time [s]

        Notes:

        References:
            A. E. Hubbard et al., “Physics and performance of the I-mode regime over an expanded operating space on Alcator C-Mod,”
            Nuclear Fusion, vol. 57, no. 12, p. 126039, Oct. 2017, doi: https://doi.org/10.1088/1741-4326/aa8570.
            ‌
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return 0.014 * I_p_MA ** 0.68 * B0 ** 0.77 * dnla20 ** 0.02 * P_loss_MW ** (-0.29)
def tau_E_I_Mode_y2(I_p: float, B0: float, P_loss: float, n_la: float) -> float:
    """
    Calculate the I-Mode confinement time scaling from Walk (equation 5.2).

    Args:
        I_p (float): Plasma current [A]
        B0 (float): Toroidal magnetic field on axis [T]
        P_loss (float): Net heating power [W]
        n_la (float): Line averaged electron density in m**-3

    Returns:
        float: I-Mode confinement time [s]

    Notes:
        Coefficient adjusted for density in 1e19 m^-3.

    References:
        J. R. Walk, "Pedestal structure and stability in high-performance plasmas on Alcator C-Mod,"
        https://dspace.mit.edu/handle/1721.1/95524, equation 5.2. 2014
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.01346 
        * B0 ** 0.768 
        * I_p_MA ** 0.685 
        * P_loss_MW ** (-0.286) 
        * dnla19 ** 0.017)



def tau_E_lang_high_density(
    I_p: float,
    B0: float,
    nd_line: float,
    P_loss: float,
    R: float,
    a: float,
    q: float,
    q_star: float,
    A: float,
    afuel: float,
    kappa_ipb: float,
) -> float:
    """Return tau E lang high density scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    qratio = q / q_star
    n_gw = 1.0e20 * I_p_MA / (sp.pi * a * a)
    nratio = nd_line / n_gw
    return (
        6.94e-7
        * I_p_MA ** 1.3678
        * B0 ** 0.12
        * nd_line ** 0.032236
        * (P_loss_MW * 1.0e6) ** (-0.74)
        * R ** 1.2345
        * kappa_ipb ** 0.37
        * A ** 2.48205
        * afuel ** 0.2
        * qratio ** 0.77
        * A ** (-0.9 * sp.log(A))
        * nratio ** (-0.22 * sp.log(nratio))
    )




def tau_E_petty08(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float
) -> float:
    """
        Calculate the beta independent dimensionless Petty08 confinement time

        Args:
            pcur (float): Plasma current [A]
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            n_la (float): Line averaged electron density in m**-3
            p_plasma_loss (float): Net Heating power [W]
            rmajor (float): Plasma major radius [m]
            kappa_ipb (float): IPB specific plasma separatrix elongation
            aspect (float): Aspect ratio

        Returns:
            float: Petty08 confinement time [s]

        Notes:
            This scaling uses the IPB defintiion of elongation, see reference for more information.

        References:
            C. C. Petty, “Sizing up plasmas using dimensionless parameters,”
            Physics of Plasmas, vol. 15, no. 8, Aug. 2008, doi: https://doi.org/10.1063/1.2961043.

            None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
            Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
            ‌
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.052
        * I_p_MA ** 0.75
        * B0 ** 0.3
        * dnla19 ** 0.32
        * P_loss_MW ** (-0.47)
        * R ** 2.09
        * kappa_ipb ** 0.88
        * A ** (-0.84)
    )




def tau_E_murari(I_p: float, R: float, kappa_ipb: float, n_la: float, B0: float, P_loss: float) -> float:
    """
        Calculate the Murari H-mode energy confinement scaling time

        Args:
            pcur (float): Plasma current [A]
            rmajor (float): Plasma major radius [m]
            kappa_ipb (float): IPB specific plasma separatrix elongation
            n_la (float): Line averaged electron density in m**-3
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            p_plasma_loss (float): Net Heating power [W]

        Returns:
            float: Murari confinement time [s]

        Notes:
            This scaling uses the IPB defintiion of elongation, see reference for more information.

        References:
            A. Murari, E. Peluso, Michela Gelfusa, I. Lupelli, and P. Gaudio, “A new approach to the formulation and validation of scaling expressions for plasma confinement in tokamaks,”
            Nuclear Fusion, vol. 55, no. 7, pp. 073009-073009, Jun. 2015, doi: https://doi.org/10.1088/0029-5515/55/7/073009.

            None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
            Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
            ‌
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.0367
        * I_p_MA ** 1.006
        * R ** 1.731
        * kappa_ipb ** 1.450
        * P_loss_MW ** (-0.735)
        * (dnla19 ** 0.448 / (1.0 + sp.exp(-9.403 * (dnla19 / B0) ** -1.365)))
    )




def tau_E_ds03(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_95: float, A: float, afuel: float
) -> float:
    """
        Calculate the DS03 beta-independent H-mode scaling confinement time

        Args:
            pcur (float): Plasma current [A]
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            n_la (float): Line averaged electron density in m**-3
            p_plasma_loss (float): Net Heating power [W]
            rmajor (float): Plasma major radius [m]
            kappa95 (float): Plasma elongation at 95% flux surface
            aspect (float): Aspect ratio
            afuel (float): Fuel atomic mass number

        Returns:
            float: DS03 beta-independent H-mode confinement time [s]

        Notes:

        References:
            T. C. Luce, C. C. Petty, and J. G. Cordey, “Application of dimensionless parameter scaling techniques to the design and interpretation of magnetic fusion experiments,”
            Plasma Physics and Controlled Fusion, vol. 50, no. 4, p. 043001, Mar. 2008,
            doi: https://doi.org/10.1088/0741-3335/50/4/043001.
            ‌
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.028
        * I_p_MA ** 0.83
        * B0 ** 0.07
        * dnla19 ** 0.49
        * P_loss_MW ** (-0.55)
        * R ** 2.11
        * kappa_95 ** 0.75
        * A ** (-0.3)
        * afuel ** 0.14
    )




def tau_E_iss04_stellarator(a: float, R: float, n_la: float, B0: float, P_loss: float, iotabar: float) -> float:
    """
        Calculate the ISS04 stellarator scaling confinement time

        Args:
            rminor (float): Plasma minor radius [m]
            rmajor (float): Plasma major radius [m]
            n_la (float): Line averaged electron density in m**-3
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            p_plasma_loss (float): Net Heating power [W]
            iotabar (float): Rotational transform

        Returns:
            float: ISS04 stellarator confinement time [s]

        Notes:

        References:
            H. Yamada et al., “Characterization of energy confinement in net-current free plasmas using the extended International Stellarator Database,”
            vol. 45, no. 12, pp. 1684-1693, Nov. 2005, doi: https://doi.org/10.1088/0029-5515/45/12/024.
            ‌
    """
    dnla19 = n_la / 1e19
    P_loss_MW = P_loss / 1e6
    return 0.134 * a ** 2.28 * R ** 0.64 * dnla19 ** 0.54 * B0 ** 0.84 * P_loss_MW ** (-0.61) * iotabar ** 0.41




def tau_E_iss95_stellarator(a: float, R: float, n_la: float, B0: float, P_loss: float, iotabar: float) -> float:
    """
        Calculate the ISS95 stellarator scaling confinement time

        Args:
            rminor (float): Plasma minor radius [m]
            rmajor (float): Plasma major radius [m]
            n_la (float): Line averaged electron density in m**-3
            b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
            p_plasma_loss (float): Net Heating power [W]
            iotabar (float): Rotational transform

        Returns:
            float: ISS95 stellarator confinement time [s]

        Notes:

        References:
            U. Stroth et al., “Energy confinement scaling from the international stellarator database,”
            vol. 36, no. 8, pp. 1063-1077, Aug. 1996, doi: https://doi.org/10.1088/0029-5515/36/8/i11.
            ‌
    """
    dnla19 = n_la / 1e19
    P_loss_MW = P_loss / 1e6
    return 0.079 * a ** 2.21 * R ** 0.65 * dnla19 ** 0.51 * B0 ** 0.83 * P_loss_MW ** (-0.59) * iotabar ** 0.4




def tau_E_iter_ipb98y4(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,4) ELMy H-mode scaling confinement time

    Args:
        pcur (float): Plasma current [A]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        n_la (float): Line averaged electron density in m**-3
        p_plasma_loss (float): Net Heating power [W]
        rmajor (float): Plasma major radius [m]
        kappa_ipb (float): IPB specific plasma separatrix elongation
        aspect (float): Aspect ratio
        afuel (float): Fuel atomic mass number

    Returns:
        float: IPB98(y,4) ELMy H-mode confinement time [s]

    Notes:
        See correction paper below for more information about the re-definition of the elongation used.

    References:
        I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
        Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.0587
        * I_p_MA ** 0.85
        * B0 ** 0.29
        * dnla19 ** 0.39
        * P_loss_MW ** (-0.70)
        * R ** 2.08
        * kappa_ipb ** 0.76
        * A ** (-0.69)
        * afuel ** 0.17
    )




def tau_E_iter_ipb98y3(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,3) ELMy H-mode scaling confinement time

    Args:
        pcur (float): Plasma current [A]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        n_la (float): Line averaged electron density in m**-3
        p_plasma_loss (float): Net Heating power [W]
        rmajor (float): Plasma major radius [m]
        kappa_ipb (float): IPB specific plasma separatrix elongation
        aspect (float): Aspect ratio
        afuel (float): Fuel atomic mass number [amu]

    Returns:
        float: IPB98(y,3) ELMy H-mode confinement time [s]

    Notes:
        See correction paper below for more information about the re-definition of the elongation used.

    References:
        I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
        Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.0564
        * I_p_MA ** 0.88
        * B0 ** 0.07
        * dnla19 ** 0.40
        * P_loss_MW ** (-0.69)
        * R ** 2.15
        * kappa_ipb ** 0.78
        * A ** (-0.64)
        * afuel ** 0.20
    )



########################################
@Relation(
    name="tau_E_iter_ipb98y2",
    output="tau_E",
    tags=("confinement", "tokamak", "hmode"),
)
def tau_E_iter_ipb98y2(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Taken from PROCESS codebase.
    Calculate the IPB98(y,2) ELMy H-mode scaling confinement time

    Args:
        I_p (float): Plasma current [A]
        B0 (float): Toroidal magnetic field on axis[T]
        n_la (float): Line averaged electron density [m**-3]
        P_loss (float): Net Heating power [W]
        R (float): Plasma major radius [m]
        kappa_ipb (float): IPB specific plasma separatrix elongation
        A (float): Aspect ratio
        afuel (float): Fuel atomic mass number

    Returns:
        float: IPB98(y,2) ELMy H-mode confinement time [s]

    Notes:
        See correction paper below for more information about the re-definition of the elongation used.

    References:
        I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
        Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    return 0.0562 * (I_p / 1e6) ** 0.93 * B0 ** 0.15 * (n_la/1e19) ** 0.41 * (P_loss/1e6)** (-0.69) * R ** 1.97 * kappa_ipb ** 0.78 * A ** (-0.58) * afuel ** 0.19
def tau_E_iter_ipb98y1(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,1) ELMy H-mode scaling confinement time

    Args:
        pcur (float): Plasma current [A]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        n_la (float): Line averaged electron density in m**-3
        p_plasma_loss (float): Net Heating power [W]
        rmajor (float): Plasma major radius [m]
        kappa_ipb (float): IPB sprcific plasma separatrix elongation
        aspect (float): Aspect ratio
        afuel (float): Fuel atomic mass number

    Returns:
        float: IPB98(y,1) ELMy H-mode confinement time [s]

    Notes:
        See correction paper below for more information about the re-definition of the elongation used.

    References:
        I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
        Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.0503
        * I_p_MA ** 0.91
        * B0 ** 0.15
        * dnla19 ** 0.44
        * P_loss_MW ** (-0.65)
        * R ** 2.05
        * kappa_ipb ** 0.72
        * A ** (-0.57)
        * afuel ** 0.13
    )



def iter_ipb98y_confinement_time(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa: float, A: float, afuel: float
) -> float:
    """Return iter ipb98y confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.0365
        * I_p_MA ** 0.97
        * B0 ** 0.08
        * dnla19 ** 0.41
        * P_loss_MW ** (-0.63)
        * R ** 1.93
        * kappa ** 0.67
        * A ** (-0.23)
        * afuel ** 0.2
    )

tau_E_ipb98y = iter_ipb98y_confinement_time


def iter_pb98py_confinement_time(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, kappa: float, A: float, afuel: float
) -> float:
    """Return iter pb98py confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.0615
        * I_p_MA ** 0.9
        * B0 ** 0.1
        * dnla19 ** 0.4
        * P_loss_MW ** (-0.66)
        * R**2
        * kappa ** 0.75
        * A ** (-0.66)
        * afuel ** 0.2
    )

tau_E_pb98py = iter_pb98py_confinement_time

def kaye_confinement_time(
    I_p: float, B0: float, kappa: float, R: float, A: float, n_la: float, afuel: float, P_loss: float
) -> float:
    """Return kaye confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.021
        * I_p_MA ** 0.81
        * B0 ** 0.14
        * kappa ** 0.7
        * R ** 2.01
        * A ** (-0.18)
        * dnla19 ** 0.47
        * afuel ** 0.25
        * P_loss_MW ** (-0.73)
    )


def valovic_elmy_confinement_time(
    I_p: float, B0: float, n_la: float, afuel: float, R: float, a: float, kappa: float, P_loss: float
) -> float:
    """Return valovic elmy confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.067
        * I_p_MA ** 0.9
        * B0 ** 0.17
        * dnla19 ** 0.45
        * afuel ** 0.05
        * R ** 1.316
        * a ** 0.79
        * kappa ** 0.56
        * P_loss_MW ** (-0.68)
    )



def iter_96p_confinement_time(
    I_p: float, B0: float, kappa_95: float, R: float, A: float, n_la: float, afuel: float, P_loss: float
) -> float:
    """Return iter 96p confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.023
        * I_p_MA ** 0.96
        * B0 ** 0.03
        * kappa_95 ** 0.64
        * R ** 1.83
        * A ** 0.06
        * dnla19 ** 0.40
        * afuel ** 0.20
        * P_loss_MW ** (-0.73)
    )

tau_E_iter96p = iter_96p_confinement_time


def tau_E_iter97L(
    I_p: float, B0: float, n_la: float, P_loss: float, R: float, eps: float, kappa_A: float, afuel: float
) -> float:
    """ITER97L scaling (density in 1e19 m^-3, eps=a/R)."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    eps_safe = eps
    return (
        0.023
        * I_p_MA ** 0.96
        * B0 ** 0.03
        * R ** 1.83
        * eps_safe ** -0.06
        * kappa_A ** 0.64
        * dnla19 ** 0.40
        * afuel ** 0.20
        * P_loss_MW ** (-0.73)
    )


def iter_h97p_elmy_confinement_time(
    I_p: float, B0: float, P_loss: float, n_la: float, R: float, A: float, kappa: float, afuel: float
) -> float:
    """Return iter h97p elmy confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.029
        * I_p_MA ** 0.90
        * B0 ** 0.20
        * P_loss_MW ** (-0.66)
        * dnla19 ** 0.40
        * R ** 2.03
        * A ** (-0.19)
        * kappa ** 0.92
        * afuel ** 0.2
    )

tau_E_iter_h97p_elmy = iter_h97p_elmy_confinement_time


def iter_h97p_confinement_time(
    I_p: float, B0: float, P_loss: float, n_la: float, R: float, A: float, kappa: float, afuel: float
) -> float:
    """Return iter h97p confinement time scaling."""
    dnla19 = n_la / 1e19
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.031
        * I_p_MA ** 0.95
        * B0 ** 0.25
        * P_loss_MW ** (-0.67)
        * dnla19 ** 0.35
        * R ** 1.92
        * A ** (-0.08)
        * kappa ** 0.63
        * afuel ** 0.42
    )

tau_E_iter_h97p = iter_h97p_confinement_time


def iter_93h_confinement_time(
    I_p: float, B0: float, P_loss: float, afuel: float, R: float, dnla20: float, A: float, kappa: float
) -> float:
    """Return iter 93h confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.036
        * I_p_MA ** 1.06
        * B0 ** 0.32
        * P_loss_MW ** (-0.67)
        * afuel ** 0.41
        * R ** 1.79
        * dnla20 ** 0.17
        * A ** 0.11
        * kappa ** 0.66
    )

tau_E_iter93h = iter_93h_confinement_time


def lackner_gottardi_stellarator_confinement_time(
    R: float, a: float, dnla20: float, B0: float, P_loss: float, q: float
) -> float:
    """Return lackner gottardi stellarator confinement time scaling."""
    P_loss_MW = P_loss / 1e6
    return 0.17 * R * a**2 * dnla20 ** 0.6 * B0 ** 0.8 * P_loss_MW ** (-0.6) * q ** 0.4



def gyro_reduced_bohm_confinement_time(B0: float, dnla20: float, P_loss: float, a: float, R: float) -> float:
    """Return gyro reduced bohm confinement time scaling."""
    P_loss_MW = P_loss / 1e6
    return 0.25 * B0 ** 0.8 * dnla20 ** 0.6 * P_loss_MW ** (-0.6) * a ** 2.4 * R ** 0.6



def sudo_et_al_confinement_time(R: float, a: float, dnla20: float, B0: float, P_loss: float) -> float:
    """Return sudo et al confinement time scaling."""
    P_loss_MW = P_loss / 1e6
    return 0.17 * R ** 0.75 * a**2 * dnla20 ** 0.69 * B0 ** 0.84 * P_loss_MW ** (-0.58)



def iter_h90p_amended_confinement_time(I_p: float, B0: float, afuel: float, R: float, P_loss: float, kappa: float) -> float:
    """Return iter h90p amended confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return 0.082 * I_p_MA ** 1.02 * B0 ** 0.15 * sp.sqrt(afuel) * R ** 1.60 / (P_loss_MW ** 0.47 * kappa ** 0.19)

tau_E_iter_h90p_amended = iter_h90p_amended_confinement_time


def riedel_h_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, afuel: float, P_loss: float
) -> float:
    """Return riedel h confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.1
        * sp.sqrt(afuel)
        * I_p_MA ** 0.884
        * R ** 1.24
        * a ** (-0.23)
        * kappa_95 ** 0.317
        * B0 ** 0.207
        * dnla20 ** 0.105
        / P_loss_MW ** 0.486
    )



def neo_kaye_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss: float
) -> float:
    """Return neo kaye confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.063
        * I_p_MA ** 1.12
        * R ** 1.3
        * a ** (-0.04)
        * kappa_95 ** 0.28
        * dnla20 ** 0.14
        * B0 ** 0.04
        / P_loss_MW ** 0.59
    )



def lackner_gottardi_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss: float
) -> float:
    """Return lackner gottardi confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    qhat = ((1.0 + kappa_95**2) * a * a * B0) / (0.4 * I_p_MA * R)
    return (
        0.12
        * I_p_MA ** 0.8
        * R ** 1.8
        * a ** 0.4
        * kappa_95
        * (1.0 + kappa_95) ** (-0.8)
        * dnla20 ** 0.6
        * qhat ** 0.4
        / P_loss_MW ** 0.6
    )



def christiansen_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss: float, afuel: float
) -> float:
    """Return christiansen confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.24
        * I_p_MA ** 0.79
        * R ** 0.56
        * a ** 1.46
        * kappa_95 ** 0.73
        * dnla20 ** 0.41
        * B0 ** 0.29
        / (P_loss_MW ** 0.79 * afuel ** 0.02)
    )



def riedel_l_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss: float
) -> float:
    """Return riedel l confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.044
        * I_p_MA ** 0.93
        * R ** 1.37
        * a ** (-0.049)
        * kappa_95 ** 0.588
        * dnla20 ** 0.078
        * B0 ** 0.152
        / P_loss_MW ** 0.537
    )



def iter_h90p_confinement_time(
    I_p: float, R: float, a: float, kappa: float, dnla20: float, B0: float, afuel: float, P_loss: float
) -> float:
    """Return iter h90p confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.064
        * I_p_MA ** 0.87
        * R ** 1.82
        * a ** (-0.12)
        * kappa ** 0.35
        * dnla20 ** 0.09
        * B0 ** 0.15
        * sp.sqrt(afuel)
        / sp.sqrt(P_loss_MW)
    )

tau_E_iter_h90p = iter_h90p_confinement_time


def kaye_big_confinement_time(
    R: float, a: float, B0: float, kappa_95: float, I_p: float, n20: float, afuel: float, P_loss: float
) -> float:
    """Return kaye big confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.105
        * sp.sqrt(R)
        * a ** 0.8
        * B0 ** 0.3
        * kappa_95 ** 0.25
        * I_p_MA ** 0.85
        * n20 ** 0.1
        * sp.sqrt(afuel)
        / sp.sqrt(P_loss_MW)
    )



def jaeri_confinement_time(
    kappa_95: float,
    a: float,
    afuel: float,
    n20: float,
    I_p: float,
    B0: float,
    R: float,
    q_star: float,
    Z_eff: float,
    P_loss: float,
) -> float:
    """Return jaeri confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    gjaeri = (
        Z_eff**0.4
        * ((15.0 - Z_eff) / 20.0) ** 0.6
        * (3.0 * q_star * (q_star + 5.0) / ((q_star + 2.0) * (q_star + 7.0))) ** 0.6
    )
    return (
        0.085 * kappa_95 * a**2 * sp.sqrt(afuel)
        + 0.069
        * n20 ** 0.6
        * I_p_MA
        * B0 ** 0.2
        * a ** 0.4
        * R ** 1.6
        * sp.sqrt(afuel)
        * gjaeri
        * kappa_95 ** 0.2
        / P_loss_MW
    )



def t10_confinement_time(
    dnla20: float,
    R: float,
    q_star: float,
    B0: float,
    a: float,
    kappa_95: float,
    P_loss: float,
    Z_eff: float,
    I_p: float,
) -> float:
    """Return t10 confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    denfac = dnla20 * R * q_star / (1.3 * B0)
    denfac = min(1.0, denfac)
    return (
        0.095
        * R
        * a
        * B0
        * sp.sqrt(kappa_95)
        * denfac
        / (P_loss_MW ** 0.4)
        * (Z_eff**2 * I_p_MA**4 / (R * a * q_star**3 * kappa_95 ** 1.5)) ** 0.08
    )

tau_E_t10 = t10_confinement_time

def goldston_confinement_time(I_p: float, R: float, a: float, kappa_95: float, afuel: float, P_loss: float) -> float:
    """Return goldston confinement time scaling."""
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return 0.037 * I_p_MA * R ** 1.75 * a ** (-0.37) * sp.sqrt(kappa_95) * sp.sqrt(afuel / 1.5) / sp.sqrt(P_loss_MW)



def rebut_lallia_confinement_time(
    a: float,
    R: float,
    kappa: float,
    afuel: float,
    I_p: float,
    Z_eff: float,
    dnla20: float,
    B0: float,
    P_loss: float,
) -> float:
    """
    Calculate the Rebut-Lallia offset linear scaling (L-mode) confinement time

    Args:
        rminor (float): Plasma minor radius [m]
        rmajor (float): Plasma major radius [m]
        kappa (float): Plasma elongation at 95% flux surface
        afuel (float): Fuel atomic mass number
        pcur (float): Plasma current [A]
        zeff (float): Effective charge
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        p_plasma_loss (float): Net Heating power [W]

    Returns:
        float: Rebut-Lallia confinement time [s]

    References:
        T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    rll = (a * a * R * kappa) ** (1.0 / 3.0)
    term1 = 1.2e-2 * I_p_MA * rll ** 1.5 / sp.sqrt(Z_eff)
    term2 = 0.146 * dnla20 ** 0.75 * sp.sqrt(I_p_MA) * sp.sqrt(B0) * rll ** 2.75 * Z_eff ** 0.25 / P_loss_MW
    return 1.65 * sp.sqrt(afuel / 2.0) * (term1 + term2)



def iter_89o_confinement_time(
    I_p: float, R: float, a: float, kappa: float, dnla20: float, B0: float, afuel: float, P_loss: float
) -> float:
    """
    Calculate the ITER Offset linear scaling - ITER 89-O (L-mode) confinement time

    Args:
        pcur (float): Plasma current [A]
        rmajor (float): Plasma major radius [m]
        rminor (float): Plasma minor radius [m]
        kappa (float): Plasma elongation
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        afuel (float): Fuel atomic mass number
        p_plasma_loss (float): Net Heating power [W]

    Returns:
        float: ITER 89-O confinement time [s]

    References:
        T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    term1 = 0.04 * I_p_MA ** 0.5 * R ** 0.3 * a ** 0.8 * kappa ** 0.6 * afuel ** 0.5
    term2 = (
        0.064
        * I_p_MA ** 0.8
        * R ** 1.6
        * a ** 0.6
        * kappa ** 0.5
        * dnla20 ** 0.6
        * B0 ** 0.35
        * afuel ** 0.2
        / P_loss_MW
    )
    return term1 + term2



def iter_89p_confinement_time(I_p: float, R: float, a: float, kappa: float, dnla20: float, B0: float, afuel: float, P_loss: float) -> float:  # noqa: E501
    """
    Calculate the ITER Power scaling - ITER 89-P (L-mode) confinement time

    Args:
        pcur (float): Plasma current [A]
        rmajor (float): Plasma major radius [m]
        rminor (float): Plasma minor radius [m]
        kappa (float): Plasma elongation
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        afuel (float): Fuel atomic mass number
        p_plasma_loss (float): Net Heating power [W]

    Returns:
        float: ITER 89-P confinement time [s]

    References:
        T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992
        N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and ITER Physics Group,
        "ITER physics design guidelines: 1989", no. No. 10. Feb. 1990.
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.048
        * I_p_MA ** 0.85
        * R ** 1.2
        * a ** 0.3
        * sp.sqrt(kappa)
        * dnla20 ** 0.1
        * B0 ** 0.2
        * sp.sqrt(afuel)
        / sp.sqrt(P_loss_MW)
    )


def tau_E_iter89P(
    I_p: float, R: float, a: float, kappa_sep: float, dnla20: float, B0: float, afuel: float, P_loss: float
) -> float:
    """
    ITER89P scaling using separatrix elongation (kappa) with density in 1e20 m^-3.

    Reference: Yushmanov et al., "Scalings for tokamak energy confinement," Nuclear Fusion 30(10), 1990 (coefficients adjusted for density units).
    
    L-mode
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.03812775526676551
        * afuel ** 0.5
        * B0 ** 0.2
        * I_p_MA ** 0.85
        * R ** 1.2
        * a ** 0.3
        * kappa_sep ** 0.5
        * dnla20 ** 0.1
        / sp.sqrt(P_loss_MW)
    )


def tau_E_iter89P_ka(
    I_p: float, R: float, a: float, kappa_A: float, dnla20: float, B0: float, afuel: float, P_loss: float
) -> float:
    """
    ITER89P scaling using areal elongation (kappa_A) with density in 1e20 m^-3.

    Reference: Yushmanov et al., "Scalings for tokamak energy confinement," Nuclear Fusion 30(10), 1990 (coefficients adjusted for density units).
    
    L-mode
    """
    I_p_MA = I_p / 1e6
    P_loss_MW = P_loss / 1e6
    return (
        0.03812775526676551
        * afuel ** 0.5
        * B0 ** 0.2
        * I_p_MA ** 0.85
        * R ** 1.2
        * a ** 0.3
        * kappa_A ** 0.5
        * dnla20 ** 0.1
        / sp.sqrt(P_loss_MW)
    )



def tau_E_mirnov(a: float, kappa_95: float, I_p: float) -> float:
    """
    Calculate the Mirnov scaling (H-mode) confinement time

    Args:
        rminor (float): Plasma minor radius [m]
        kappa95 (float): Plasma elongation at 95% flux surface
        pcur (float): Plasma current [A]

    Returns:
        float: Mirnov scaling confinement time [s]

    References:
        N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and ITER Physics Group,
        "ITER physics design guidelines: 1989", no. No. 10. Feb. 1990.
    """
    I_p_MA = I_p / 1e6
    return 0.2 * a * sp.sqrt(kappa_95) * I_p_MA




def tau_E_neo_alcator(dene20: float, a: float, R: float, q_star: float) -> float:
    """
    Calculate the Nec-Alcator(NA) OH scaling confinement time

    Args:
        dene20 (float): Volume averaged electron density in units of 10**20 m**-3
        rminor (float): Plasma minor radius [m]
        rmajor (float): Plasma major radius [m]
        qstar (float): Equivalent cylindrical edge safety factor

    Returns:
        float: Neo-Alcator confinement time [s]

    References:
        N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and ITER Physics Group,
        "ITER physics design guidelines: 1989", no. No. 10. Feb. 1990.
    """
    return 0.07 * dene20 * a * R * R * q_star
