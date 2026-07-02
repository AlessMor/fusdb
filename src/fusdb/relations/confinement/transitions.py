"""Confinement transition threshold relations."""

from typing import Any

from fusdb import relation


@relation(
    name='L-H transition threshold power',
    tags=('confinement', 'h_mode', 'constraint'),
    outputs='P_LH',
)
def lh_transition_power(n_avg: float, B0: float, A_p: float) -> Any:
    """Return the L-H transition threshold power using a Martin-2008 style scaling.

    Args:
        n_avg: Line-averaged density [1/m^3].
        B0: Toroidal magnetic field [T].
        A_p: Plasma surface area [m^2].

    Returns:
        L-H transition threshold power [W].
    """
    n20 = n_avg / 1e20
    # P_LH [MW] = 0.0488 * n20^0.717 * B0^0.803 * A_p^0.941
    return 1e6 * 0.0488 * (n20 ** 0.717) * (B0 ** 0.803) * (A_p ** 0.941)


# L-I (L-mode to I-mode) transition threshold power. cfspopcon bundles three
# scalings behind an enum; following the lambda_q pattern they are imported as
# separate relations (all output P_LI_thresh), selected via the
# P_LI_thresh.default_relation gate (HubbardNF17 default) or explicit include.
# cfspopcon expresses I_p in MA and n_e in 1e19 m^-3; outputs are in MW -> W.


@relation(
    name="L-I transition threshold power HubbardNF17",
    tags=("confinement", "i_mode", "tokamak"),
    outputs="P_LI_thresh",
)
def calc_LI_transition_threshold_power_HubbardNF17(n_e_avg, B0, A_p, confinement_threshold_scalar=1.0):
    """L-I threshold power, Hubbard NF 2017 scaling (Fig 6 of :cite:`hubbard_threshold_2017`).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    n19 = n_e_avg / 1.0e19
    return 1.0e6 * (0.162 * (n19 / 10.0) * (B0**0.262) * A_p) * confinement_threshold_scalar


@relation(
    name="L-I transition threshold power AUG",
    tags=("confinement", "i_mode", "tokamak"),
    outputs="P_LI_thresh",
)
def calc_LI_transition_threshold_power_AUG(n_e_avg, B0, A_p, confinement_threshold_scalar=1.0):
    """L-I threshold power, AUG scaling (:cite:`ryter_i-mode_2016`, :cite:`Happel_2017`).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    n19 = n_e_avg / 1.0e19
    return 1.0e6 * (0.14 * (n19 / 10.0) * (B0 / 2.4) ** 0.39 * A_p) * confinement_threshold_scalar


@relation(
    name="L-I transition threshold power HubbardNF12",
    tags=("confinement", "i_mode", "tokamak"),
    outputs="P_LI_thresh",
)
def calc_LI_transition_threshold_power_HubbardNF12(I_p, n_e_avg, confinement_threshold_scalar=1.0):
    """L-I threshold power, Hubbard NF 2012 scaling (Fig 5 of :cite:`hubbard_threshold_2012`).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    plasma_current = I_p / 1.0e6
    n19 = n_e_avg / 1.0e19
    return 1.0e6 * (2.11 * plasma_current**0.94 * ((n19 / 10.0) ** 0.65)) * confinement_threshold_scalar


@relation(
    name="Ratio of P_SOL to P_LI",
    tags=("confinement", "i_mode", "tokamak"),
    outputs="ratio_of_P_SOL_to_P_LI",
)
def calc_ratio_P_LI(P_sep, P_LI_thresh):
    """Ratio of the power crossing the separatrix to the L-I threshold power.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_sep / P_LI_thresh