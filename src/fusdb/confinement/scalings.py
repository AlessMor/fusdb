"""Energy confinement time scalings defined as independent relations.
Most of them are taken as-is from the PROCESS code (UKAEA)."""

from __future__ import annotations

import math

from fusdb.relations_util import require_nonzero
from fusdb.relations_values import PRIORITY_RELATION, Relation


# --- Selected scaling functions with full docstrings -------------------------
def tau_E_neo_alcator(dene20: float, a: float, R: float, q_star: float) -> float:
    """
    Calculate the Nec-Alcator(NA) OH scaling confinement time

    Parameters:
    dene20 (float): Volume averaged electron density in units of 10**20 m**-3
    rminor (float): Plasma minor radius [m]
    rmajor (float): Plasma major radius [m]
    qstar (float): Equivalent cylindrical edge safety factor

    Returns:
    float: Neo-Alcator confinement time [s]

    References:
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and ITER Physics Group,
         "ITER physics design guidelines: 1989", no. No. 10. Feb. 1990.
    """
    return 0.07 * dene20 * a * R * R * q_star


tau_E_neo_alcator_relation = Relation(
    "tau_E neo_alcator",
    ("tau_E_neo_alcator", "dene20", "a", "R", "q_star"),
    lambda v: v["tau_E_neo_alcator"] - tau_E_neo_alcator(v["dene20"], v["a"], v["R"], v["q_star"]),
    solve_for=("tau_E_neo_alcator",),
    priority=PRIORITY_RELATION,
    initial_guesses={"tau_E_neo_alcator": lambda v: tau_E_neo_alcator(v["dene20"], v["a"], v["R"], v["q_star"])},
)


def tau_E_mirnov(a: float, kappa_95: float, I_p: float) -> float:
    """
    Calculate the Mirnov scaling (H-mode) confinement time

    Parameters:
    rminor (float): Plasma minor radius [m]
    kappa95 (float): Plasma elongation at 95% flux surface
    pcur (float): Plasma current [MA]

    Returns:
    float: Mirnov scaling confinement time [s]

    References:
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and ITER Physics Group,
         "ITER physics design guidelines: 1989", no. No. 10. Feb. 1990.
    """
    return 0.2 * a * math.sqrt(kappa_95) * I_p


tau_E_mirnov_relation = Relation(
    "tau_E mirnov",
    ("tau_E_mirnov", "a", "kappa_95", "I_p"),
    lambda v: v["tau_E_mirnov"] - tau_E_mirnov(v["a"], v["kappa_95"], v["I_p"]),
    solve_for=("tau_E_mirnov",),
    priority=PRIORITY_RELATION,
    initial_guesses={"tau_E_mirnov": lambda v: tau_E_mirnov(v["a"], v["kappa_95"], v["I_p"])},
)


def iter_89p_confinement_time(I_p: float, R: float, a: float, kappa: float, dnla20: float, B0: float, afuel: float, P_loss_MW: float) -> float:  # noqa: E501
    """
    Calculate the ITER Power scaling - ITER 89-P (L-mode) confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    rmajor (float): Plasma major radius [m]
    rminor (float): Plasma minor radius [m]
    kappa (float): Plasma elongation
    dnla20 (float): Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    afuel (float): Fuel atomic mass number
    p_plasma_loss_mw (float): Net Heating power [MW]

    Returns:
    float: ITER 89-P confinement time [s]

    References:
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria)and ITER Physics Group,
          "ITER physics design guidelines: 1989", no. No. 10. Feb. 1990.
    """
    return (
        0.048
        * I_p ** 0.85
        * R ** 1.2
        * a ** 0.3
        * math.sqrt(kappa)
        * dnla20 ** 0.1
        * B0 ** 0.2
        * math.sqrt(afuel)
        / math.sqrt(P_loss_MW)
    )

tau_E_iter_89p_relation = Relation(
    "tau_E iter89p",
    ("tau_E_iter_89p", "I_p", "R", "a", "kappa", "dnla20", "B0", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_iter_89p"] - iter_89p_confinement_time(v["I_p"], v["R"], v["a"], v["kappa"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_iter_89p",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_89p": lambda v: iter_89p_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]
        )
    },
)


def iter_89o_confinement_time(
    I_p: float, R: float, a: float, kappa: float, dnla20: float, B0: float, afuel: float, P_loss_MW: float
) -> float:
    """
    Calculate the ITER Offset linear scaling - ITER 89-O (L-mode) confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    rmajor (float): Plasma major radius [m]
    rminor (float): Plasma minor radius [m]
    kappa (float): Plasma elongation
    dnla20 (float): Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    afuel (float): Fuel atomic mass number
    p_plasma_loss_mw (float): Net Heating power [MW]

    Returns:
    float: ITER 89-O confinement time [s]

    References:
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992
    """
    term1 = 0.04 * I_p ** 0.5 * R ** 0.3 * a ** 0.8 * kappa ** 0.6 * afuel ** 0.5
    term2 = (
        0.064
        * I_p ** 0.8
        * R ** 1.6
        * a ** 0.6
        * kappa ** 0.5
        * dnla20 ** 0.6
        * B0 ** 0.35
        * afuel ** 0.2
        / P_loss_MW
    )
    return term1 + term2

tau_E_iter_89o_relation = Relation(
    "tau_E iter89o",
    ("tau_E_iter_89o", "I_p", "R", "a", "kappa", "dnla20", "B0", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_iter_89o"] - iter_89o_confinement_time(v["I_p"], v["R"], v["a"], v["kappa"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_iter_89o",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_89o": lambda v: iter_89o_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]
        )
    },
)


def rebut_lallia_confinement_time(
    a: float,
    R: float,
    kappa: float,
    afuel: float,
    I_p: float,
    Z_eff: float,
    dnla20: float,
    B0: float,
    P_loss_MW: float,
) -> float:
    """
    Calculate the Rebut-Lallia offset linear scaling (L-mode) confinement time

    Parameters:
    rminor (float): Plasma minor radius [m]
    rmajor (float): Plasma major radius [m]
    kappa (float): Plasma elongation at 95% flux surface
    afuel (float): Fuel atomic mass number
    pcur (float): Plasma current [MA]
    zeff (float): Effective charge
    dnla20 (float): Line averaged electron density in units of 10**20 m**-3
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    p_plasma_loss_mw (float): Net Heating power [MW]

    Returns:
    float: Rebut-Lallia confinement time [s]

    References:
        - T.C.Hender et.al., 'Physics Assesment of the European Reactor Study', AEA FUS 172, 1992
    """
    rll = (a * a * R * kappa) ** (1.0 / 3.0)
    term1 = 1.2e-2 * I_p * rll ** 1.5 / math.sqrt(Z_eff)
    term2 = 0.146 * dnla20 ** 0.75 * math.sqrt(I_p) * math.sqrt(B0) * rll ** 2.75 * Z_eff ** 0.25 / P_loss_MW
    return 1.65 * math.sqrt(afuel / 2.0) * (term1 + term2)

tau_E_rebut_lallia_relation = Relation(
    "tau_E rebut_lallia",
    ("tau_E_rebut_lallia", "a", "R", "kappa", "afuel", "I_p", "Z_eff", "dnla20", "B0", "P_loss_MW"),
    lambda v: v["tau_E_rebut_lallia"] - rebut_lallia_confinement_time(v["a"], v["R"], v["kappa"], v["afuel"], v["I_p"], v["Z_eff"], v["dnla20"], v["B0"], v["P_loss_MW"]),
    solve_for=("tau_E_rebut_lallia",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_rebut_lallia": lambda v: rebut_lallia_confinement_time(
            v["a"], v["R"], v["kappa"], v["afuel"], v["I_p"], v["Z_eff"], v["dnla20"], v["B0"], v["P_loss_MW"]
        )
    },
)


def goldston_confinement_time(I_p: float, R: float, a: float, kappa_95: float, afuel: float, P_loss_MW: float) -> float:
    return 0.037 * I_p * R ** 1.75 * a ** (-0.37) * math.sqrt(kappa_95) * math.sqrt(afuel / 1.5) / math.sqrt(P_loss_MW)

tau_E_goldston_relation = Relation(
    "tau_E goldston",
    ("tau_E_goldston", "I_p", "R", "a", "kappa_95", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_goldston"] - goldston_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_goldston",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_goldston": lambda v: goldston_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["afuel"], v["P_loss_MW"])
    },
)


def t10_confinement_time(
    dnla20: float,
    R: float,
    q_star: float,
    B0: float,
    a: float,
    kappa_95: float,
    P_loss_MW: float,
    Z_eff: float,
    I_p: float,
) -> float:
    denfac = dnla20 * R * q_star / (1.3 * B0)
    denfac = min(1.0, denfac)
    return (
        0.095
        * R
        * a
        * B0
        * math.sqrt(kappa_95)
        * denfac
        / (P_loss_MW ** 0.4)
        * (Z_eff**2 * I_p**4 / (R * a * q_star**3 * kappa_95 ** 1.5)) ** 0.08
    )

tau_E_t10 = t10_confinement_time
tau_E_t10_relation = Relation(
    "tau_E t10",
    ("tau_E_t10", "dnla20", "R", "q_star", "B0", "a", "kappa_95", "P_loss_MW", "Z_eff", "I_p"),
    lambda v: v["tau_E_t10"] - tau_E_t10(v["dnla20"], v["R"], v["q_star"], v["B0"], v["a"], v["kappa_95"], v["P_loss_MW"], v["Z_eff"], v["I_p"]),
    solve_for=("tau_E_t10",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_t10": lambda v: tau_E_t10(v["dnla20"], v["R"], v["q_star"], v["B0"], v["a"], v["kappa_95"], v["P_loss_MW"], v["Z_eff"], v["I_p"])
    },
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
    P_loss_MW: float,
) -> float:
    gjaeri = (
        Z_eff**0.4
        * ((15.0 - Z_eff) / 20.0) ** 0.6
        * (3.0 * q_star * (q_star + 5.0) / ((q_star + 2.0) * (q_star + 7.0))) ** 0.6
    )
    return (
        0.085 * kappa_95 * a**2 * math.sqrt(afuel)
        + 0.069
        * n20 ** 0.6
        * I_p
        * B0 ** 0.2
        * a ** 0.4
        * R ** 1.6
        * math.sqrt(afuel)
        * gjaeri
        * kappa_95 ** 0.2
        / P_loss_MW
    )

tau_E_jaeri_relation = Relation(
    "tau_E jaeri",
    ("tau_E_jaeri", "kappa_95", "a", "afuel", "n20", "I_p", "B0", "R", "q_star", "Z_eff", "P_loss_MW"),
    lambda v: v["tau_E_jaeri"] - jaeri_confinement_time(v["kappa_95"], v["a"], v["afuel"], v["n20"], v["I_p"], v["B0"], v["R"], v["q_star"], v["Z_eff"], v["P_loss_MW"]),
    solve_for=("tau_E_jaeri",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_jaeri": lambda v: jaeri_confinement_time(
            v["kappa_95"],
            v["a"],
            v["afuel"],
            v["n20"],
            v["I_p"],
            v["B0"],
            v["R"],
            v["q_star"],
            v["Z_eff"],
            v["P_loss_MW"],
        )
    },
)


def kaye_big_confinement_time(
    R: float, a: float, B0: float, kappa_95: float, I_p: float, n20: float, afuel: float, P_loss_MW: float
) -> float:
    return (
        0.105
        * math.sqrt(R)
        * a ** 0.8
        * B0 ** 0.3
        * kappa_95 ** 0.25
        * I_p ** 0.85
        * n20 ** 0.1
        * math.sqrt(afuel)
        / math.sqrt(P_loss_MW)
    )

tau_E_kaye_big_relation = Relation(
    "tau_E kaye_big",
    ("tau_E_kaye_big", "R", "a", "B0", "kappa_95", "I_p", "n20", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_kaye_big"] - kaye_big_confinement_time(v["R"], v["a"], v["B0"], v["kappa_95"], v["I_p"], v["n20"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_kaye_big",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_kaye_big": lambda v: kaye_big_confinement_time(
            v["R"], v["a"], v["B0"], v["kappa_95"], v["I_p"], v["n20"], v["afuel"], v["P_loss_MW"]
        )
    },
)


def iter_h90p_confinement_time(
    I_p: float, R: float, a: float, kappa: float, dnla20: float, B0: float, afuel: float, P_loss_MW: float
) -> float:
    return (
        0.064
        * I_p ** 0.87
        * R ** 1.82
        * a ** (-0.12)
        * kappa ** 0.35
        * dnla20 ** 0.09
        * B0 ** 0.15
        * math.sqrt(afuel)
        / math.sqrt(P_loss_MW)
    )

tau_E_iter_h90p = iter_h90p_confinement_time
tau_E_iter_h90p_relation = Relation(
    "tau_E iter_h90p",
    ("tau_E_iter_h90p", "I_p", "R", "a", "kappa", "dnla20", "B0", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_iter_h90p"] - tau_E_iter_h90p(v["I_p"], v["R"], v["a"], v["kappa"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_iter_h90p",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_h90p": lambda v: tau_E_iter_h90p(v["I_p"], v["R"], v["a"], v["kappa"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"])
    },
)


def riedel_l_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss_MW: float
) -> float:
    return (
        0.044
        * I_p ** 0.93
        * R ** 1.37
        * a ** (-0.049)
        * kappa_95 ** 0.588
        * dnla20 ** 0.078
        * B0 ** 0.152
        / P_loss_MW ** 0.537
    )

tau_E_riedel_l_relation = Relation(
    "tau_E riedel_l",
    ("tau_E_riedel_l", "I_p", "R", "a", "kappa_95", "dnla20", "B0", "P_loss_MW"),
    lambda v: v["tau_E_riedel_l"] - riedel_l_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"]),
    solve_for=("tau_E_riedel_l",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_riedel_l": lambda v: riedel_l_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"]
        )
    },
)


def christiansen_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss_MW: float, afuel: float
) -> float:
    return (
        0.24
        * I_p ** 0.79
        * R ** 0.56
        * a ** 1.46
        * kappa_95 ** 0.73
        * dnla20 ** 0.41
        * B0 ** 0.29
        / (P_loss_MW ** 0.79 * afuel ** 0.02)
    )

tau_E_christiansen_relation = Relation(
    "tau_E christiansen",
    ("tau_E_christiansen", "I_p", "R", "a", "kappa_95", "dnla20", "B0", "P_loss_MW", "afuel"),
    lambda v: v["tau_E_christiansen"] - christiansen_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"], v["afuel"]),
    solve_for=("tau_E_christiansen",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_christiansen": lambda v: christiansen_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"], v["afuel"]
        )
    },
)


def lackner_gottardi_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss_MW: float
) -> float:
    qhat = ((1.0 + kappa_95**2) * a * a * B0) / (0.4 * I_p * R)
    return (
        0.12
        * I_p ** 0.8
        * R ** 1.8
        * a ** 0.4
        * kappa_95
        * (1.0 + kappa_95) ** (-0.8)
        * dnla20 ** 0.6
        * qhat ** 0.4
        / P_loss_MW ** 0.6
    )

tau_E_lackner_gottardi_relation = Relation(
    "tau_E lackner_gottardi",
    ("tau_E_lackner_gottardi", "I_p", "R", "a", "kappa_95", "dnla20", "B0", "P_loss_MW"),
    lambda v: v["tau_E_lackner_gottardi"] - lackner_gottardi_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"]),
    solve_for=("tau_E_lackner_gottardi",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_lackner_gottardi": lambda v: lackner_gottardi_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"]
        )
    },
)


def neo_kaye_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, P_loss_MW: float
) -> float:
    return (
        0.063
        * I_p ** 1.12
        * R ** 1.3
        * a ** (-0.04)
        * kappa_95 ** 0.28
        * dnla20 ** 0.14
        * B0 ** 0.04
        / P_loss_MW ** 0.59
    )

tau_E_neo_kaye_relation = Relation(
    "tau_E neo_kaye",
    ("tau_E_neo_kaye", "I_p", "R", "a", "kappa_95", "dnla20", "B0", "P_loss_MW"),
    lambda v: v["tau_E_neo_kaye"] - neo_kaye_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"]),
    solve_for=("tau_E_neo_kaye",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_neo_kaye": lambda v: neo_kaye_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["P_loss_MW"]
        )
    },
)


def riedel_h_confinement_time(
    I_p: float, R: float, a: float, kappa_95: float, dnla20: float, B0: float, afuel: float, P_loss_MW: float
) -> float:
    return (
        0.1
        * math.sqrt(afuel)
        * I_p ** 0.884
        * R ** 1.24
        * a ** (-0.23)
        * kappa_95 ** 0.317
        * B0 ** 0.207
        * dnla20 ** 0.105
        / P_loss_MW ** 0.486
    )

tau_E_riedel_h_relation = Relation(
    "tau_E riedel_h",
    ("tau_E_riedel_h", "I_p", "R", "a", "kappa_95", "dnla20", "B0", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_riedel_h"] - riedel_h_confinement_time(v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_riedel_h",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_riedel_h": lambda v: riedel_h_confinement_time(
            v["I_p"], v["R"], v["a"], v["kappa_95"], v["dnla20"], v["B0"], v["afuel"], v["P_loss_MW"]
        )
    },
)


def iter_h90p_amended_confinement_time(I_p: float, B0: float, afuel: float, R: float, P_loss_MW: float, kappa: float) -> float:
    return 0.082 * I_p ** 1.02 * B0 ** 0.15 * math.sqrt(afuel) * R ** 1.60 / (P_loss_MW ** 0.47 * kappa ** 0.19)

tau_E_iter_h90p_amended = iter_h90p_amended_confinement_time
tau_E_iter_h90p_amended_relation = Relation(
    "tau_E iter_h90p_amended",
    ("tau_E_iter_h90p_amended", "I_p", "B0", "afuel", "R", "P_loss_MW", "kappa"),
    lambda v: v["tau_E_iter_h90p_amended"] - tau_E_iter_h90p_amended(v["I_p"], v["B0"], v["afuel"], v["R"], v["P_loss_MW"], v["kappa"]),
    solve_for=("tau_E_iter_h90p_amended",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_h90p_amended": lambda v: tau_E_iter_h90p_amended(v["I_p"], v["B0"], v["afuel"], v["R"], v["P_loss_MW"], v["kappa"])
    },
)


def sudo_et_al_confinement_time(R: float, a: float, dnla20: float, B0: float, P_loss_MW: float) -> float:
    return 0.17 * R ** 0.75 * a**2 * dnla20 ** 0.69 * B0 ** 0.84 * P_loss_MW ** (-0.58)

tau_E_sudo_relation = Relation(
    "tau_E sudo",
    ("tau_E_sudo", "R", "a", "dnla20", "B0", "P_loss_MW"),
    lambda v: v["tau_E_sudo"] - sudo_et_al_confinement_time(v["R"], v["a"], v["dnla20"], v["B0"], v["P_loss_MW"]),
    solve_for=("tau_E_sudo",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_sudo": lambda v: sudo_et_al_confinement_time(v["R"], v["a"], v["dnla20"], v["B0"], v["P_loss_MW"])
    },
)


def gyro_reduced_bohm_confinement_time(B0: float, dnla20: float, P_loss_MW: float, a: float, R: float) -> float:
    return 0.25 * B0 ** 0.8 * dnla20 ** 0.6 * P_loss_MW ** (-0.6) * a ** 2.4 * R ** 0.6

tau_E_grb_relation = Relation(
    "tau_E gyro_reduced_bohm",
    ("tau_E_grb", "B0", "dnla20", "P_loss_MW", "a", "R"),
    lambda v: v["tau_E_grb"] - gyro_reduced_bohm_confinement_time(v["B0"], v["dnla20"], v["P_loss_MW"], v["a"], v["R"]),
    solve_for=("tau_E_grb",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_grb": lambda v: gyro_reduced_bohm_confinement_time(v["B0"], v["dnla20"], v["P_loss_MW"], v["a"], v["R"])
    },
)


def lackner_gottardi_stellarator_confinement_time(
    R: float, a: float, dnla20: float, B0: float, P_loss_MW: float, q: float
) -> float:
    return 0.17 * R * a**2 * dnla20 ** 0.6 * B0 ** 0.8 * P_loss_MW ** (-0.6) * q ** 0.4

tau_E_lg_stell_relation = Relation(
    "tau_E lackner_gottardi_stellarator",
    ("tau_E_lg_stell", "R", "a", "dnla20", "B0", "P_loss_MW", "q"),
    lambda v: v["tau_E_lg_stell"] - lackner_gottardi_stellarator_confinement_time(v["R"], v["a"], v["dnla20"], v["B0"], v["P_loss_MW"], v["q"]),
    solve_for=("tau_E_lg_stell",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_lg_stell": lambda v: lackner_gottardi_stellarator_confinement_time(
            v["R"], v["a"], v["dnla20"], v["B0"], v["P_loss_MW"], v["q"]
        )
    },
)


def iter_93h_confinement_time(
    I_p: float, B0: float, P_loss_MW: float, afuel: float, R: float, dnla20: float, A: float, kappa: float
) -> float:
    return (
        0.036
        * I_p ** 1.06
        * B0 ** 0.32
        * P_loss_MW ** (-0.67)
        * afuel ** 0.41
        * R ** 1.79
        * dnla20 ** 0.17
        * A ** 0.11
        * kappa ** 0.66
    )

tau_E_iter93h = iter_93h_confinement_time
tau_E_iter93h_relation = Relation(
    "tau_E iter93h",
    ("tau_E_iter93h", "I_p", "B0", "P_loss_MW", "afuel", "R", "dnla20", "A", "kappa"),
    lambda v: v["tau_E_iter93h"] - tau_E_iter93h(v["I_p"], v["B0"], v["P_loss_MW"], v["afuel"], v["R"], v["dnla20"], v["A"], v["kappa"]),
    solve_for=("tau_E_iter93h",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter93h": lambda v: tau_E_iter93h(v["I_p"], v["B0"], v["P_loss_MW"], v["afuel"], v["R"], v["dnla20"], v["A"], v["kappa"])
    },
)


def iter_h97p_confinement_time(
    I_p: float, B0: float, P_loss_MW: float, dnla19: float, R: float, A: float, kappa: float, afuel: float
) -> float:
    return (
        0.031
        * I_p ** 0.95
        * B0 ** 0.25
        * P_loss_MW ** (-0.67)
        * dnla19 ** 0.35
        * R ** 1.92
        * A ** (-0.08)
        * kappa ** 0.63
        * afuel ** 0.42
    )

tau_E_iter_h97p = iter_h97p_confinement_time
tau_E_iter_h97p_relation = Relation(
    "tau_E iter_h97p",
    ("tau_E_iter_h97p", "I_p", "B0", "P_loss_MW", "dnla19", "R", "A", "kappa", "afuel"),
    lambda v: v["tau_E_iter_h97p"] - tau_E_iter_h97p(v["I_p"], v["B0"], v["P_loss_MW"], v["dnla19"], v["R"], v["A"], v["kappa"], v["afuel"]),
    solve_for=("tau_E_iter_h97p",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_h97p": lambda v: tau_E_iter_h97p(v["I_p"], v["B0"], v["P_loss_MW"], v["dnla19"], v["R"], v["A"], v["kappa"], v["afuel"])
    },
)


def iter_h97p_elmy_confinement_time(
    I_p: float, B0: float, P_loss_MW: float, dnla19: float, R: float, A: float, kappa: float, afuel: float
) -> float:
    return (
        0.029
        * I_p ** 0.90
        * B0 ** 0.20
        * P_loss_MW ** (-0.66)
        * dnla19 ** 0.40
        * R ** 2.03
        * A ** (-0.19)
        * kappa ** 0.92
        * afuel ** 0.2
    )

tau_E_iter_h97p_elmy = iter_h97p_elmy_confinement_time
tau_E_iter_h97p_elmy_relation = Relation(
    "tau_E iter_h97p_elmy",
    ("tau_E_iter_h97p_elmy", "I_p", "B0", "P_loss_MW", "dnla19", "R", "A", "kappa", "afuel"),
    lambda v: v["tau_E_iter_h97p_elmy"] - tau_E_iter_h97p_elmy(v["I_p"], v["B0"], v["P_loss_MW"], v["dnla19"], v["R"], v["A"], v["kappa"], v["afuel"]),
    solve_for=("tau_E_iter_h97p_elmy",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_h97p_elmy": lambda v: tau_E_iter_h97p_elmy(v["I_p"], v["B0"], v["P_loss_MW"], v["dnla19"], v["R"], v["A"], v["kappa"], v["afuel"])
    },
)


def iter_96p_confinement_time(
    I_p: float, B0: float, kappa_95: float, R: float, A: float, dnla19: float, afuel: float, P_loss_MW: float
) -> float:
    return (
        0.023
        * I_p ** 0.96
        * B0 ** 0.03
        * kappa_95 ** 0.64
        * R ** 1.83
        * A ** 0.06
        * dnla19 ** 0.40
        * afuel ** 0.20
        * P_loss_MW ** (-0.73)
    )

tau_E_iter96p = iter_96p_confinement_time
tau_E_iter96p_relation = Relation(
    "tau_E iter96p",
    ("tau_E_iter96p", "I_p", "B0", "kappa_95", "R", "A", "dnla19", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_iter96p"] - tau_E_iter96p(v["I_p"], v["B0"], v["kappa_95"], v["R"], v["A"], v["dnla19"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_iter96p",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter96p": lambda v: tau_E_iter96p(v["I_p"], v["B0"], v["kappa_95"], v["R"], v["A"], v["dnla19"], v["afuel"], v["P_loss_MW"])
    },
)


def valovic_elmy_confinement_time(
    I_p: float, B0: float, dnla19: float, afuel: float, R: float, a: float, kappa: float, P_loss_MW: float
) -> float:
    return (
        0.067
        * I_p ** 0.9
        * B0 ** 0.17
        * dnla19 ** 0.45
        * afuel ** 0.05
        * R ** 1.316
        * a ** 0.79
        * kappa ** 0.56
        * P_loss_MW ** (-0.68)
    )

tau_E_valovic_relation = Relation(
    "tau_E valovic",
    ("tau_E_valovic", "I_p", "B0", "dnla19", "afuel", "R", "a", "kappa", "P_loss_MW"),
    lambda v: v["tau_E_valovic"] - valovic_elmy_confinement_time(v["I_p"], v["B0"], v["dnla19"], v["afuel"], v["R"], v["a"], v["kappa"], v["P_loss_MW"]),
    solve_for=("tau_E_valovic",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_valovic": lambda v: valovic_elmy_confinement_time(
            v["I_p"], v["B0"], v["dnla19"], v["afuel"], v["R"], v["a"], v["kappa"], v["P_loss_MW"]
        )
    },
)


def kaye_confinement_time(
    I_p: float, B0: float, kappa: float, R: float, A: float, dnla19: float, afuel: float, P_loss_MW: float
) -> float:
    return (
        0.021
        * I_p ** 0.81
        * B0 ** 0.14
        * kappa ** 0.7
        * R ** 2.01
        * A ** (-0.18)
        * dnla19 ** 0.47
        * afuel ** 0.25
        * P_loss_MW ** (-0.73)
    )

tau_E_kaye_relation = Relation(
    "tau_E kaye_1998",
    ("tau_E_kaye", "I_p", "B0", "kappa", "R", "A", "dnla19", "afuel", "P_loss_MW"),
    lambda v: v["tau_E_kaye"] - kaye_confinement_time(v["I_p"], v["B0"], v["kappa"], v["R"], v["A"], v["dnla19"], v["afuel"], v["P_loss_MW"]),
    solve_for=("tau_E_kaye",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_kaye": lambda v: kaye_confinement_time(v["I_p"], v["B0"], v["kappa"], v["R"], v["A"], v["dnla19"], v["afuel"], v["P_loss_MW"])
    },
)

def iter_pb98py_confinement_time(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa: float, A: float, afuel: float
) -> float:
    return (
        0.0615
        * I_p ** 0.9
        * B0 ** 0.1
        * dnla19 ** 0.4
        * P_loss_MW ** (-0.66)
        * R**2
        * kappa ** 0.75
        * A ** (-0.66)
        * afuel ** 0.2
    )

tau_E_pb98py = iter_pb98py_confinement_time
tau_E_pb98py_relation = Relation(
    "tau_E pb98py",
    ("tau_E_pb98py", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa", "A", "afuel"),
    lambda v: v["tau_E_pb98py"] - tau_E_pb98py(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa"], v["A"], v["afuel"]),
    solve_for=("tau_E_pb98py",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_pb98py": lambda v: tau_E_pb98py(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa"], v["A"], v["afuel"])
    },
)

def iter_ipb98y_confinement_time(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa: float, A: float, afuel: float
) -> float:
    return (
        0.0365
        * I_p ** 0.97
        * B0 ** 0.08
        * dnla19 ** 0.41
        * P_loss_MW ** (-0.63)
        * R ** 1.93
        * kappa ** 0.67
        * A ** (-0.23)
        * afuel ** 0.2
    )

tau_E_ipb98y = iter_ipb98y_confinement_time
tau_E_ipb98y_relation = Relation(
    "tau_E ipb98y",
    ("tau_E_ipb98y", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa", "A", "afuel"),
    lambda v: v["tau_E_ipb98y"] - tau_E_ipb98y(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa"], v["A"], v["afuel"]),
    solve_for=("tau_E_ipb98y",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_ipb98y": lambda v: tau_E_ipb98y(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa"], v["A"], v["afuel"])
    },
)


def tau_E_iter_ipb98y1(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,1) ELMy H-mode scaling confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    dnla19 (float): Line averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw (float): Net Heating power [MW]
    rmajor (float): Plasma major radius [m]
    kappa_ipb (float): IPB sprcific plasma separatrix elongation
    aspect (float): Aspect ratio
    afuel (float): Fuel atomic mass number

    Returns:
    float: IPB98(y,1) ELMy H-mode confinement time [s]

    Notes:
        - See correction paper below for more information about the re-definition of the elongation used.

    References:
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
          Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    return (
        0.0503
        * I_p ** 0.91
        * B0 ** 0.15
        * dnla19 ** 0.44
        * P_loss_MW ** (-0.65)
        * R ** 2.05
        * kappa_ipb ** 0.72
        * A ** (-0.57)
        * afuel ** 0.13
    )

tau_E_iter_ipb98y1_relation = Relation(
    "tau_E ipb98y1",
    ("tau_E_iter_ipb98y1", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A", "afuel"),
    lambda v: v["tau_E_iter_ipb98y1"] - tau_E_iter_ipb98y1(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]),
    solve_for=("tau_E_iter_ipb98y1",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_ipb98y1": lambda v: tau_E_iter_ipb98y1(
            v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]
        )
    },
)


def tau_E_iter_ipb98y2(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,2) ELMy H-mode scaling confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    dnla19 (float): Line averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw (float): Net Heating power [MW]
    rmajor (float): Plasma major radius [m]
    kappa_ipb (float): IPB specific plasma separatrix elongation
    aspect (float): Aspect ratio
    afuel (float): Fuel atomic mass number

    Returns:
    float: IPB98(y,2) ELMy H-mode confinement time [s]

    Notes:
        - See correction paper below for more information about the re-definition of the elongation used.

    References:
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
          Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    return (
        0.0562
        * I_p ** 0.93
        * B0 ** 0.15
        * dnla19 ** 0.41
        * P_loss_MW ** (-0.69)
        * R ** 1.97
        * kappa_ipb ** 0.78
        * A ** (-0.58)
        * afuel ** 0.19
    )

tau_E_iter_ipb98y2_relation = Relation(
    "tau_E ipb98y2",
    ("tau_E_iter_ipb98y2", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A", "afuel"),
    lambda v: v["tau_E_iter_ipb98y2"] - tau_E_iter_ipb98y2(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]),
    solve_for=("tau_E_iter_ipb98y2",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_ipb98y2": lambda v: tau_E_iter_ipb98y2(
            v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]
        )
    },
)


def tau_E_iter_ipb98y3(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,3) ELMy H-mode scaling confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    dnla19 (float): Line averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw (float): Net Heating power [MW]
    rmajor (float): Plasma major radius [m]
    kappa_ipb (float): IPB specific plasma separatrix elongation
    aspect (float): Aspect ratio
    afuel (float): Fuel atomic mass number

    Returns:
    float: IPB98(y,3) ELMy H-mode confinement time [s]

    Notes:
        - See correction paper below for more information about the re-definition of the elongation used.

    References:
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
          Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    return (
        0.0564
        * I_p ** 0.88
        * B0 ** 0.07
        * dnla19 ** 0.40
        * P_loss_MW ** (-0.69)
        * R ** 2.15
        * kappa_ipb ** 0.78
        * A ** (-0.64)
        * afuel ** 0.20
    )

tau_E_iter_ipb98y3_relation = Relation(
    "tau_E ipb98y3",
    ("tau_E_iter_ipb98y3", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A", "afuel"),
    lambda v: v["tau_E_iter_ipb98y3"] - tau_E_iter_ipb98y3(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]),
    solve_for=("tau_E_iter_ipb98y3",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_ipb98y3": lambda v: tau_E_iter_ipb98y3(
            v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]
        )
    },
)


def tau_E_iter_ipb98y4(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    """
    Calculate the IPB98(y,4) ELMy H-mode scaling confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    dnla19 (float): Line averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw (float): Net Heating power [MW]
    rmajor (float): Plasma major radius [m]
    kappa_ipb (float): IPB specific plasma separatrix elongation
    aspect (float): Aspect ratio
    afuel (float): Fuel atomic mass number

    Returns:
    float: IPB98(y,4) ELMy H-mode confinement time [s]

    Notes:
        - See correction paper below for more information about the re-definition of the elongation used.

    References:
        - I. P. E. G. on C. Transport, I. P. E. G. on C. Database, and I. P. B. Editors, “Chapter 2: Plasma confinement and transport,”
        Nuclear Fusion, vol. 39, no. 12, pp. 2175-2249, Dec. 1999, doi: https://doi.org/10.1088/0029-5515/39/12/302.

        - None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
          Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    """
    return (
        0.0587
        * I_p ** 0.85
        * B0 ** 0.29
        * dnla19 ** 0.39
        * P_loss_MW ** (-0.70)
        * R ** 2.08
        * kappa_ipb ** 0.76
        * A ** (-0.69)
        * afuel ** 0.17
    )


tau_E_iter_ipb98y4_relation = Relation(
    "tau_E ipb98y4",
    ("tau_E_iter_ipb98y4", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A", "afuel"),
    lambda v: v["tau_E_iter_ipb98y4"] - tau_E_iter_ipb98y4(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]),
    solve_for=("tau_E_iter_ipb98y4",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iter_ipb98y4": lambda v: tau_E_iter_ipb98y4(
            v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]
        )
    },
)


def tau_E_iss95_stellarator(a: float, R: float, dnla19: float, B0: float, P_loss_MW: float, iotabar: float) -> float:
    """
        Calculate the ISS95 stellarator scaling confinement time

        Parameters:
        rminor (float): Plasma minor radius [m]
        rmajor (float): Plasma major radius [m]
        dnla19 (float): Line averaged electron density in units of 10**19 m**-3
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        p_plasma_loss_mw (float): Net Heating power [MW]
        iotabar (float): Rotational transform

        Returns:
        float: ISS95 stellarator confinement time [s]

        Notes:

        References:
            - U. Stroth et al., “Energy confinement scaling from the international stellarator database,”
              vol. 36, no. 8, pp. 1063-1077, Aug. 1996, doi: https://doi.org/10.1088/0029-5515/36/8/i11.
    ‌
    """
    return 0.079 * a ** 2.21 * R ** 0.65 * dnla19 ** 0.51 * B0 ** 0.83 * P_loss_MW ** (-0.59) * iotabar ** 0.4


tau_E_iss95_stellarator_relation = Relation(
    "tau_E iss95_stellarator",
    ("tau_E_iss95_stellarator", "a", "R", "dnla19", "B0", "P_loss_MW", "iotabar"),
    lambda v: v["tau_E_iss95_stellarator"] - tau_E_iss95_stellarator(v["a"], v["R"], v["dnla19"], v["B0"], v["P_loss_MW"], v["iotabar"]),
    solve_for=("tau_E_iss95_stellarator",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iss95_stellarator": lambda v: tau_E_iss95_stellarator(v["a"], v["R"], v["dnla19"], v["B0"], v["P_loss_MW"], v["iotabar"])
    },
)


def tau_E_iss04_stellarator(a: float, R: float, dnla19: float, B0: float, P_loss_MW: float, iotabar: float) -> float:
    """
        Calculate the ISS04 stellarator scaling confinement time

        Parameters:
        rminor (float): Plasma minor radius [m]
        rmajor (float): Plasma major radius [m]
        dnla19 (float): Line averaged electron density in units of 10**19 m**-3
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        p_plasma_loss_mw (float): Net Heating power [MW]
        iotabar (float): Rotational transform

        Returns:
        float: ISS04 stellarator confinement time [s]

        Notes:

        References:
            - H. Yamada et al., “Characterization of energy confinement in net-current free plasmas using the extended International Stellarator Database,”
              vol. 45, no. 12, pp. 1684-1693, Nov. 2005, doi: https://doi.org/10.1088/0029-5515/45/12/024.
    ‌
    """
    return 0.134 * a ** 2.28 * R ** 0.64 * dnla19 ** 0.54 * B0 ** 0.84 * P_loss_MW ** (-0.61) * iotabar ** 0.41


tau_E_iss04_stellarator_relation = Relation(
    "tau_E iss04_stellarator",
    ("tau_E_iss04_stellarator", "a", "R", "dnla19", "B0", "P_loss_MW", "iotabar"),
    lambda v: v["tau_E_iss04_stellarator"] - tau_E_iss04_stellarator(v["a"], v["R"], v["dnla19"], v["B0"], v["P_loss_MW"], v["iotabar"]),
    solve_for=("tau_E_iss04_stellarator",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_iss04_stellarator": lambda v: tau_E_iss04_stellarator(v["a"], v["R"], v["dnla19"], v["B0"], v["P_loss_MW"], v["iotabar"])
    },
)


def tau_E_ds03(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_95: float, A: float, afuel: float
) -> float:
    """
        Calculate the DS03 beta-independent H-mode scaling confinement time

        Parameters:
        pcur (float): Plasma current [MA]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        dnla19 (float): Line averaged electron density in units of 10**19 m**-3
        p_plasma_loss_mw (float): Net Heating power [MW]
        rmajor (float): Plasma major radius [m]
        kappa95 (float): Plasma elongation at 95% flux surface
        aspect (float): Aspect ratio
        afuel (float): Fuel atomic mass number

        Returns:
        float: DS03 beta-independent H-mode confinement time [s]

        Notes:

        References:
            - T. C. Luce, C. C. Petty, and J. G. Cordey, “Application of dimensionless parameter scaling techniques to the design and interpretation of magnetic fusion experiments,”
             Plasma Physics and Controlled Fusion, vol. 50, no. 4, p. 043001, Mar. 2008,
             doi: https://doi.org/10.1088/0741-3335/50/4/043001.
    ‌
    """
    return (
        0.028
        * I_p ** 0.83
        * B0 ** 0.07
        * dnla19 ** 0.49
        * P_loss_MW ** (-0.55)
        * R ** 2.11
        * kappa_95 ** 0.75
        * A ** (-0.3)
        * afuel ** 0.14
    )


tau_E_ds03_relation = Relation(
    "tau_E ds03",
    ("tau_E_ds03", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_95", "A", "afuel"),
    lambda v: v["tau_E_ds03"] - tau_E_ds03(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_95"], v["A"], v["afuel"]),
    solve_for=("tau_E_ds03",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_ds03": lambda v: tau_E_ds03(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_95"], v["A"], v["afuel"])
    },
)


def tau_E_murari(I_p: float, R: float, kappa_ipb: float, dnla19: float, B0: float, P_loss_MW: float) -> float:
    """
        Calculate the Murari H-mode energy confinement scaling time

        Parameters:
        pcur (float): Plasma current [MA]
        rmajor (float): Plasma major radius [m]
        kappa_ipb (float): IPB specific plasma separatrix elongation
        dnla19 (float): Line averaged electron density in units of 10**19 m**-3
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        p_plasma_loss_mw (float): Net Heating power [MW]

        Returns:
        float: Murari confinement time [s]

        Notes:
            - This scaling uses the IPB defintiion of elongation, see reference for more information.

        References:
            - A. Murari, E. Peluso, Michela Gelfusa, I. Lupelli, and P. Gaudio, “A new approach to the formulation and validation of scaling expressions for plasma confinement in tokamaks,”
             Nuclear Fusion, vol. 55, no. 7, pp. 073009-073009, Jun. 2015, doi: https://doi.org/10.1088/0029-5515/55/7/073009.

            - None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
              Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    ‌
    """
    return (
        0.0367
        * I_p ** 1.006
        * R ** 1.731
        * kappa_ipb ** 1.450
        * P_loss_MW ** (-0.735)
        * (dnla19 ** 0.448 / (1.0 + math.exp(-9.403 * (dnla19 / B0) ** -1.365)))
    )


tau_E_murari_relation = Relation(
    "tau_E murari",
    ("tau_E_murari", "I_p", "R", "kappa_ipb", "dnla19", "B0", "P_loss_MW"),
    lambda v: v["tau_E_murari"] - tau_E_murari(v["I_p"], v["R"], v["kappa_ipb"], v["dnla19"], v["B0"], v["P_loss_MW"]),
    solve_for=("tau_E_murari",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_murari": lambda v: tau_E_murari(v["I_p"], v["R"], v["kappa_ipb"], v["dnla19"], v["B0"], v["P_loss_MW"])
    },
)


def tau_E_petty08(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float
) -> float:
    """
        Calculate the beta independent dimensionless Petty08 confinement time

        Parameters:
        pcur (float): Plasma current [MA]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        dnla19 (float): Line averaged electron density in units of 10**19 m**-3
        p_plasma_loss_mw (float): Net Heating power [MW]
        rmajor (float): Plasma major radius [m]
        kappa_ipb (float): IPB specific plasma separatrix elongation
        aspect (float): Aspect ratio

        Returns:
        float: Petty08 confinement time [s]

        Notes:
            - This scaling uses the IPB defintiion of elongation, see reference for more information.

        References:
            - C. C. Petty, “Sizing up plasmas using dimensionless parameters,”
            Physics of Plasmas, vol. 15, no. 8, Aug. 2008, doi: https://doi.org/10.1063/1.2961043.

            - None Otto Kardaun, N. K. Thomsen, and None Alexander Chudnovskiy, “Corrections to a sequence of papers in Nuclear Fusion,”
            Nuclear Fusion, vol. 48, no. 9, pp. 099801-099801, Aug. 2008, doi: https://doi.org/10.1088/0029-5515/48/9/099801.
    ‌
    """
    return (
        0.052
        * I_p ** 0.75
        * B0 ** 0.3
        * dnla19 ** 0.32
        * P_loss_MW ** (-0.47)
        * R ** 2.09
        * kappa_ipb ** 0.88
        * A ** (-0.84)
    )


tau_E_petty08_relation = Relation(
    "tau_E petty08",
    ("tau_E_petty08", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A"),
    lambda v: v["tau_E_petty08"] - tau_E_petty08(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"]),
    solve_for=("tau_E_petty08",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_petty08": lambda v: tau_E_petty08(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"])
    },
)


def tau_E_lang_high_density(
    I_p: float,
    B0: float,
    nd_line: float,
    P_loss_MW: float,
    R: float,
    a: float,
    q: float,
    q_star: float,
    A: float,
    afuel: float,
    kappa_ipb: float,
) -> float:
    qratio = q / require_nonzero(q_star, "q_star", "Lang scaling")
    n_gw = 1.0e14 * I_p / (math.pi * a * a)
    nratio = nd_line / n_gw
    return (
        6.94e-7
        * I_p ** 1.3678
        * B0 ** 0.12
        * nd_line ** 0.032236
        * (P_loss_MW * 1.0e6) ** (-0.74)
        * R ** 1.2345
        * kappa_ipb ** 0.37
        * A ** 2.48205
        * afuel ** 0.2
        * qratio ** 0.77
        * A ** (-0.9 * math.log(A))
        * nratio ** (-0.22 * math.log(nratio))
    )


tau_E_lang_high_density_relation = Relation(
    "tau_E lang_high_density",
    ("tau_E_lang_high_density", "I_p", "B0", "nd_line", "P_loss_MW", "R", "a", "q", "q_star", "A", "afuel", "kappa_ipb"),
    lambda v: v["tau_E_lang_high_density"] - tau_E_lang_high_density(
        v["I_p"],
        v["B0"],
        v["nd_line"],
        v["P_loss_MW"],
        v["R"],
        v["a"],
        v["q"],
        v["q_star"],
        v["A"],
        v["afuel"],
        v["kappa_ipb"],
    ),
    solve_for=("tau_E_lang_high_density",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_lang_high_density": lambda v: tau_E_lang_high_density(
            v["I_p"],
            v["B0"],
            v["nd_line"],
            v["P_loss_MW"],
            v["R"],
            v["a"],
            v["q"],
            v["q_star"],
            v["A"],
            v["afuel"],
            v["kappa_ipb"],
        )
    },
)


def tau_E_hubbard_nominal(I_p: float, B0: float, dnla20: float, P_loss_MW: float) -> float:
    """
        Calculate the Hubbard 2017 I-mode confinement time scaling - nominal

        Parameters:
        pcur (float): Plasma current [MA]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3
        p_plasma_loss_mw (float): Net Heating power [MW]

        Returns:
        float: Hubbard confinement time [s]

        Notes:

        References:
            - A. E. Hubbard et al., “Physics and performance of the I-mode regime over an expanded operating space on Alcator C-Mod,”
            Nuclear Fusion, vol. 57, no. 12, p. 126039, Oct. 2017, doi: https://doi.org/10.1088/1741-4326/aa8570.
    ‌
    """
    return 0.014 * I_p ** 0.68 * B0 ** 0.77 * dnla20 ** 0.02 * P_loss_MW ** (-0.29)


tau_E_hubbard_nominal_relation = Relation(
    "tau_E hubbard_nominal",
    ("tau_E_hubbard_nominal", "I_p", "B0", "dnla20", "P_loss_MW"),
    lambda v: v["tau_E_hubbard_nominal"] - tau_E_hubbard_nominal(v["I_p"], v["B0"], v["dnla20"], v["P_loss_MW"]),
    solve_for=("tau_E_hubbard_nominal",),
    priority=PRIORITY_RELATION,
    initial_guesses={"tau_E_hubbard_nominal": lambda v: tau_E_hubbard_nominal(v["I_p"], v["B0"], v["dnla20"], v["P_loss_MW"])},
)


def tau_E_hubbard_lower(I_p: float, B0: float, dnla20: float, P_loss_MW: float) -> float:
    """
        Calculate the Hubbard 2017 I-mode confinement time scaling - lower

        Parameters:
        pcur (float): Plasma current [MA]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3
        p_plasma_loss_mw (float): Net Heating power [MW]

        Returns:
        float: Hubbard confinement time [s]

        Notes:

        References:
            - A. E. Hubbard et al., “Physics and performance of the I-mode regime over an expanded operating space on Alcator C-Mod,”
            Nuclear Fusion, vol. 57, no. 12, p. 126039, Oct. 2017, doi: https://doi.org/10.1088/1741-4326/aa8570.
    ‌
    """
    return 0.014 * I_p ** 0.60 * B0 ** 0.70 * dnla20 ** (-0.03) * P_loss_MW ** (-0.33)


tau_E_hubbard_lower_relation = Relation(
    "tau_E hubbard_lower",
    ("tau_E_hubbard_lower", "I_p", "B0", "dnla20", "P_loss_MW"),
    lambda v: v["tau_E_hubbard_lower"] - tau_E_hubbard_lower(v["I_p"], v["B0"], v["dnla20"], v["P_loss_MW"]),
    solve_for=("tau_E_hubbard_lower",),
    priority=PRIORITY_RELATION,
    initial_guesses={"tau_E_hubbard_lower": lambda v: tau_E_hubbard_lower(v["I_p"], v["B0"], v["dnla20"], v["P_loss_MW"])},
)


def tau_E_hubbard_upper(I_p: float, B0: float, dnla20: float, P_loss_MW: float) -> float:
    """
        Calculate the Hubbard 2017 I-mode confinement time scaling - upper

        Parameters:
        pcur (float): Plasma current [MA]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3
        p_plasma_loss_mw (float): Net Heating power [MW]

        Returns:
        float: Hubbard confinement time [s]

        Notes:

        References:
            - A. E. Hubbard et al., “Physics and performance of the I-mode regime over an expanded operating space on Alcator C-Mod,”
            Nuclear Fusion, vol. 57, no. 12, p. 126039, Oct. 2017, doi: https://doi.org/10.1088/1741-4326/aa8570.
    ‌
    """
    return 0.014 * I_p ** 0.76 * B0 ** 0.84 * dnla20 ** 0.07 * P_loss_MW ** (-0.25)


tau_E_hubbard_upper_relation = Relation(
    "tau_E hubbard_upper",
    ("tau_E_hubbard_upper", "I_p", "B0", "dnla20", "P_loss_MW"),
    lambda v: v["tau_E_hubbard_upper"] - tau_E_hubbard_upper(v["I_p"], v["B0"], v["dnla20"], v["P_loss_MW"]),
    solve_for=("tau_E_hubbard_upper",),
    priority=PRIORITY_RELATION,
    initial_guesses={"tau_E_hubbard_upper": lambda v: tau_E_hubbard_upper(v["I_p"], v["B0"], v["dnla20"], v["P_loss_MW"])},
)


def tau_E_menard_nstx(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    return (
        0.095
        * I_p ** 0.57
        * B0 ** 1.08
        * dnla19 ** 0.44
        * P_loss_MW ** (-0.73)
        * R ** 1.97
        * kappa_ipb ** 0.78
        * A ** (-0.58)
        * afuel ** 0.19
    )


tau_E_menard_nstx_relation = Relation(
    "tau_E menard_nstx",
    ("tau_E_menard_nstx", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A", "afuel"),
    lambda v: v["tau_E_menard_nstx"] - tau_E_menard_nstx(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]),
    solve_for=("tau_E_menard_nstx",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_menard_nstx": lambda v: tau_E_menard_nstx(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"])
    },
)


def tau_E_menard_nstx_petty08_hybrid(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, kappa_ipb: float, A: float, afuel: float
) -> float:
    invA = 1.0 / A
    if invA <= 0.4:
        return tau_E_petty08(I_p, B0, dnla19, P_loss_MW, R, kappa_ipb, A)
    if invA >= 0.6:
        return tau_E_menard_nstx(I_p, B0, dnla19, P_loss_MW, R, kappa_ipb, A, afuel)
    w = (invA - 0.4) / (0.6 - 0.4)
    return w * tau_E_menard_nstx(I_p, B0, dnla19, P_loss_MW, R, kappa_ipb, A, afuel) + (1 - w) * tau_E_petty08(I_p, B0, dnla19, P_loss_MW, R, kappa_ipb, A)


tau_E_menard_nstx_petty08_hybrid_relation = Relation(
    "tau_E menard_nstx_petty08_hybrid",
    ("tau_E_menard_nstx_petty08_hybrid", "I_p", "B0", "dnla19", "P_loss_MW", "R", "kappa_ipb", "A", "afuel"),
    lambda v: v["tau_E_menard_nstx_petty08_hybrid"] - tau_E_menard_nstx_petty08_hybrid(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]),
    solve_for=("tau_E_menard_nstx_petty08_hybrid",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_menard_nstx_petty08_hybrid": lambda v: tau_E_menard_nstx_petty08_hybrid(
            v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["kappa_ipb"], v["A"], v["afuel"]
        )
    },
)


def tau_E_nstx_gyro_bohm(I_p: float, B0: float, P_loss_MW: float, R: float, dnla20: float) -> float:
    """
        Calculate the NSTX gyro-Bohm confinement time

        Parameters:
        pcur (float): Plasma current [MA]
        b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
        p_plasma_loss_mw (float): Net Heating power [MW]
        rmajor (float): Plasma major radius [m]
        dnla20 (float): Line averaged electron density in units of 10**20 m**-3

        Returns:
        float: NSTX gyro-Bohm confinement time [s]

        Notes:

        References:
            - P. F. Buxton, L. Connor, A. E. Costley, Mikhail Gryaznevich, and S. McNamara,
            “On the energy confinement time in spherical tokamaks: implications for the design of pilot plants and fusion reactors,”
            vol. 61, no. 3, pp. 035006-035006, Jan. 2019, doi: https://doi.org/10.1088/1361-6587/aaf7e5.
    ‌
    """
    return 0.21 * I_p ** 0.54 * B0 ** 0.91 * P_loss_MW ** (-0.38) * R ** 2.14 * dnla20 ** (-0.05)


tau_E_nstx_gyro_bohm_relation = Relation(
    "tau_E nstx_gyro_bohm",
    ("tau_E_nstx_gyro_bohm", "I_p", "B0", "P_loss_MW", "R", "dnla20"),
    lambda v: v["tau_E_nstx_gyro_bohm"] - tau_E_nstx_gyro_bohm(v["I_p"], v["B0"], v["P_loss_MW"], v["R"], v["dnla20"]),
    solve_for=("tau_E_nstx_gyro_bohm",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_nstx_gyro_bohm": lambda v: tau_E_nstx_gyro_bohm(v["I_p"], v["B0"], v["P_loss_MW"], v["R"], v["dnla20"])
    },
)


def tau_E_itpa20(
    I_p: float, B0: float, dnla19: float, P_loss_MW: float, R: float, delta: float, kappa_ipb: float, eps: float, aion: float
) -> float:
    """
    Calculate the ITPA20 Issue #3164 confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    dnla19 (float): Central line-averaged electron density in units of 10**19 m**-3
    p_plasma_loss_mw (float): Thermal power lost due to transport through the LCFS [MW]
    rmajor (float): Plasma major radius [m]
    triang (float): Triangularity
    kappa_ipb (float): IPB specific plasma separatrix elongation
    eps (float): Inverse aspect ratio
    aion (float): Average mass of all ions (amu)

    Returns:
    float: ITPA20 confinement time [s]

    Notes:
        - Mass term is the effective mass of the plasma, so we assume the total ion mass here
        - This scaling uses the IPB defintiion of elongation, see reference for more information.

    References:
        - G. Verdoolaege et al., “The updated ITPA global H-mode confinement database: description and analysis,”
          Nuclear Fusion, vol. 61, no. 7, pp. 076006-076006, Jan. 2021, doi: https://doi.org/10.1088/1741-4326/abdb91.
    """
    return (
        0.053
        * I_p ** 0.98
        * B0 ** 0.22
        * dnla19 ** 0.24
        * P_loss_MW ** (-0.669)
        * R ** 1.71
        * (1 + delta) ** 0.36
        * kappa_ipb ** 0.8
        * eps ** 0.35
        * aion ** 0.2
    )


tau_E_itpa20_relation = Relation(
    "tau_E itpa20",
    ("tau_E_itpa20", "I_p", "B0", "dnla19", "P_loss_MW", "R", "delta", "kappa_ipb", "eps", "aion"),
    lambda v: v["tau_E_itpa20"] - tau_E_itpa20(v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["delta"], v["kappa_ipb"], v["eps"], v["aion"]),
    solve_for=("tau_E_itpa20",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_itpa20": lambda v: tau_E_itpa20(
            v["I_p"], v["B0"], v["dnla19"], v["P_loss_MW"], v["R"], v["delta"], v["kappa_ipb"], v["eps"], v["aion"]
        )
    },
)


def tau_E_itpa20_il(
    I_p: float, B0: float, P_loss_MW: float, dnla19: float, aion: float, R: float, delta: float, kappa_ipb: float
) -> float:
    """
    Calculate the ITPA20-IL Issue #1852 confinement time

    Parameters:
    pcur (float): Plasma current [MA]
    b_plasma_toroidal_on_axis (float): Toroidal magnetic field [T]
    p_plasma_loss_mw (float): Thermal power lost due to transport through the LCFS [MW]
    dnla19 (float): Central line-averaged electron density in units of 10**19 m**-3
    aion (float): Average mass of all ions (amu)
    rmajor (float): Plasma major radius [m]
    triang (float): Triangularity
    kappa_ipb (float): IPB specific plasma separatrix elongation

    Returns:
    float: ITPA20-IL confinement time [s]

    Notes:
        - Mass term is the effective mass of the plasma, so we assume the total ion mass here
        - This scaling uses the IPB defintiion of elongation, see reference for more information.

    References:
        - T. Luda et al., “Validation of a full-plasma integrated modeling approach on ASDEX Upgrade,”
        Nuclear Fusion, vol. 61, no. 12, pp. 126048-126048, Nov. 2021, doi: https://doi.org/10.1088/1741-4326/ac3293.
    """
    return (
        0.067
        * I_p ** 1.29
        * B0 ** -0.13
        * P_loss_MW ** (-0.644)
        * dnla19 ** 0.15
        * aion ** 0.3
        * R ** 1.19
        * (1 + delta) ** 0.56
        * kappa_ipb ** 0.67
    )


tau_E_itpa20_il_relation = Relation(
    "tau_E itpa20_il",
    ("tau_E_itpa20_il", "I_p", "B0", "P_loss_MW", "dnla19", "aion", "R", "delta", "kappa_ipb"),
    lambda v: v["tau_E_itpa20_il"] - tau_E_itpa20_il(v["I_p"], v["B0"], v["P_loss_MW"], v["dnla19"], v["aion"], v["R"], v["delta"], v["kappa_ipb"]),
    solve_for=("tau_E_itpa20_il",),
    priority=PRIORITY_RELATION,
    initial_guesses={
        "tau_E_itpa20_il": lambda v: tau_E_itpa20_il(
            v["I_p"], v["B0"], v["P_loss_MW"], v["dnla19"], v["aion"], v["R"], v["delta"], v["kappa_ipb"]
        )
    },
)


# --- Relation definitions -----------------------------------------------------
TOKAMAK_CONFINEMENT_RELATIONS: tuple[Relation, ...] = (
    tau_E_neo_alcator_relation,
    tau_E_mirnov_relation,
    tau_E_iter_89p_relation,
    tau_E_iter_89o_relation,
    tau_E_rebut_lallia_relation,
    tau_E_goldston_relation,
    tau_E_t10_relation,
    tau_E_jaeri_relation,
    tau_E_kaye_big_relation,
    tau_E_iter_h90p_relation,
    tau_E_riedel_l_relation,
    tau_E_christiansen_relation,
    tau_E_lackner_gottardi_relation,
    tau_E_neo_kaye_relation,
    tau_E_riedel_h_relation,
    tau_E_iter_h90p_amended_relation,
    tau_E_sudo_relation,
    tau_E_grb_relation,
    tau_E_iter93h_relation,
    tau_E_iter_h97p_relation,
    tau_E_iter_h97p_elmy_relation,
    tau_E_iter96p_relation,
    tau_E_valovic_relation,
    tau_E_kaye_relation,
    tau_E_pb98py_relation,
    tau_E_ipb98y_relation,
    tau_E_iter_ipb98y1_relation,
    tau_E_iter_ipb98y2_relation,
    tau_E_iter_ipb98y3_relation,
    tau_E_iter_ipb98y4_relation,
    tau_E_ds03_relation,
    tau_E_murari_relation,
    tau_E_petty08_relation,
    tau_E_lang_high_density_relation,
    tau_E_hubbard_nominal_relation,
    tau_E_hubbard_lower_relation,
    tau_E_hubbard_upper_relation,
    tau_E_menard_nstx_relation,
    tau_E_menard_nstx_petty08_hybrid_relation,
    tau_E_nstx_gyro_bohm_relation,
    tau_E_itpa20_relation,
    tau_E_itpa20_il_relation,
)

STELLARATOR_CONFINEMENT_RELATIONS: tuple[Relation, ...] = (
    tau_E_lg_stell_relation,
    tau_E_iss95_stellarator_relation,
    tau_E_iss04_stellarator_relation,
)

CONFINEMENT_RELATIONS: tuple[Relation, ...] = (
    *TOKAMAK_CONFINEMENT_RELATIONS,
    *STELLARATOR_CONFINEMENT_RELATIONS,
)

CONFINEMENT_RELATIONS_BY_NAME: dict[str, Relation] = {rel.variables[0]: rel for rel in CONFINEMENT_RELATIONS}

__all__ = [
    "TOKAMAK_CONFINEMENT_RELATIONS",
    "STELLARATOR_CONFINEMENT_RELATIONS",
    "CONFINEMENT_RELATIONS",
    "CONFINEMENT_RELATIONS_BY_NAME",
]
