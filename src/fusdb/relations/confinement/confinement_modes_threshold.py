"""Confinement-mode transition threshold relations.

This module holds the confinement-mode threshold POWERS and the per-mode
CERTIFIERS that read them. Threshold producers are regime-neutral, so every
threshold is available whichever mode is being tested; the confinement-time
scalings stay mode-tagged.

Certifiers are independent, declaratively-discovered conditions: the regime
driver finds a mode's certifiers by tag and requires ALL of them, so a new
discriminant is added by writing one tagged relation and needs no driver change.
A mode is ADMISSIBLE when its own solve's relations hold AND its certifiers hold.

The forward (``P_LH``) and back (``P_HL``) transition powers differ, so the
h_mode and l_mode certifiers OVERLAP in the band between them -- both modes are
admissible there and the declared mode, fusdb's stand-in for plasma history,
decides. That is the whole hysteresis model; it needs no special case anywhere.

The confinement modes are transport states (L / H / I), not heating methods.
There is deliberately no Ohmic "mode" here: ohmic names how a plasma is heated,
and an ohmic discharge may be either L-mode or H-mode.  See the
``confinement_mode`` group in ``allowed_tags.yaml``.
"""

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(
    name='L-H transition threshold power',
    tags=('confinement', 'constraint'),
    outputs='P_LH',
)
def lh_transition_power(n_avg: float, B0: float, A_p: float) -> Any:
    """Return the L-H transition threshold power using a Martin-2008 style scaling.

    Regime-neutral (no confinement-mode tag) so ``P_LH`` is available in L-mode
    as well as H-mode; this lets the reactor-level regime verification compare
    ``P_sep`` against the threshold regardless of the candidate regime.

    Args:
        n_avg: Line-averaged density [1/m^3].
        B0: Toroidal magnetic field [T].
        A_p: Plasma surface area [m^2].

    Returns:
        L-H transition threshold power [W].
    """
    n20 = n_avg / 1e20
    # P_LH [MW] = 0.0488 * n20^0.717 * B0^0.803 * A_p^0.941
    # Superseded as the default by `lh_martin08_aspect_nominal` (PROCESS's own
    # default): this form drops Martin's ion-mass correction `2/afuel` and the
    # Takizuka aspect correction, and evaluates the fit at the volume-averaged
    # density rather than the line average it was fitted to.  Kept, gated, as
    # the bare-coefficient Martin form.
    return 1e6 * 0.0488 * (n20 ** 0.717) * (B0 ** 0.803) * (A_p ** 0.941)


@relation(
    name="H-L back-transition threshold power",
    tags=("confinement", "constraint"),
    outputs="P_HL",
)
def hl_back_transition_power(P_LH: float, f_HL_hysteresis: float = 0.7) -> Any:
    """H-L back-transition power as a fraction of the forward L-H threshold.

    An edge transport barrier, once formed, is sustained below the power needed
    to create it, so the back-transition sits below the forward threshold and the
    band between them is bistable.  DIII-D back-transition measurements give
    ``P_HL/P_LH`` in the range 0.35-0.70, and ITER studies suggest the ratio may
    need to be as low as ~0.5.

    A plain fraction of ``P_LH`` is the reduced-model form: it carries no
    independent physics beyond the forward threshold.  Replacing it with a real
    back-transition model -- or an edge-state criterion -- is a matter of
    supplying ``P_HL`` or whitelisting a different producer; nothing else changes,
    because the h_mode certifier reads the variable, not the formula.
    """
    return f_HL_hysteresis * P_LH


# Opt-in regime guards. Tagged ``confinement_mode_threshold`` (allowed_tags.yaml ``internal``
# group), which no reactor declares, so they are never picked up by automatic
# tag selection -- verify includes them by name when checking regime consistency.
# Each is an outputless, checked-only (``enforce=False``) guard: it is never a
# solver residual, and its verify status tells the Reactor whether the assumed
# regime survived the solve.  Each guard also carries its confinement-mode tag,
# which is how the Reactor discovers the guards of one regime -- there is no
# name-based table anywhere.
_GUARD_TAGS = ("confinement", "confinement_mode_threshold")


@relation(name="H-mode sustainment (P_sep >= P_HL)", tags=(*_GUARD_TAGS, "h_mode"), enforce=False)
def h_mode_sustainment(P_sep: float, P_HL: float) -> Any:
    """Certifier that holds while an H-mode solve stays above the H-L
    BACK-transition power; violated once ``P_sep`` falls below ``P_HL``.

    Deliberately NOT the complement of :func:`l_mode_sustainment`, which uses the
    forward threshold ``P_LH``.  Since ``P_HL < P_LH``, both certifiers hold in
    the band between them: both modes are admissible there, which is what makes
    the point genuinely bistable, and the declared confinement mode -- fusdb's
    stand-in for plasma history -- decides which branch is occupied.  See
    ``P_HL`` in variables.yaml.
    """
    scale = np.maximum(np.maximum(np.abs(P_sep), np.abs(P_HL)), 1.0)
    return np.maximum(P_HL - P_sep, 0.0) / scale


@relation(name="L-mode sustainment (P_sep <= P_LH)", tags=(*_GUARD_TAGS, "l_mode"), enforce=False)
def l_mode_sustainment(P_sep: float, P_LH: float) -> Any:
    """Certifier that holds while an L-mode solve stays below the FORWARD L-H
    threshold; violated once ``P_sep`` exceeds ``P_LH``.

    Not the complement of :func:`h_mode_sustainment`, which uses the lower
    back-transition power ``P_HL``: the two overlap in the hysteresis band.
    """
    scale = np.maximum(np.maximum(np.abs(P_sep), np.abs(P_LH)), 1.0)
    return np.maximum(P_sep - P_LH, 0.0) / scale


@relation(name="L-mode sustainment (P_sep <= P_LI_thresh)", tags=(*_GUARD_TAGS, "l_mode"), enforce=False)
def l_mode_below_li_sustainment(P_sep: float, P_LI_thresh: float) -> Any:
    """L-mode branch guard for the L-I bifurcation; violated once ``P_sep``
    exceeds the L-I transition threshold."""
    scale = np.maximum(np.maximum(np.abs(P_sep), np.abs(P_LI_thresh)), 1.0)
    return np.maximum(P_sep - P_LI_thresh, 0.0) / scale


@relation(name="I-mode sustainment (P_sep >= P_LI_thresh)", tags=(*_GUARD_TAGS, "i_mode"), enforce=False)
def i_mode_sustainment(P_sep: float, P_LI_thresh: float) -> Any:
    """Guard that holds while an I-mode solve stays above the L-I threshold."""
    scale = np.maximum(np.maximum(np.abs(P_sep), np.abs(P_LI_thresh)), 1.0)
    return np.maximum(P_LI_thresh - P_sep, 0.0) / scale


# L-I (L-mode to I-mode) transition threshold power. cfspopcon bundles three
# scalings behind an enum; following the lambda_q pattern they are imported as
# separate relations (all output P_LI_thresh), selected via the
# P_LI_thresh.default_relation gate or an explicit include.
# cfspopcon expresses I_p in MA and n_e in 1e19 m^-3; outputs are in MW -> W.
# NOTE these three evaluate the fits at the VOLUME-averaged density, which is
# cfspopcon's substitution -- Hubbard fits the LINE average.  The default is the
# PROCESS-derived `L-I threshold Hubbard-2017`, further down, which uses n_la.


@relation(
    name="L-I transition threshold power HubbardNF17",
    tags=("confinement", "tokamak"),
    outputs="P_LI_thresh",
)
def calc_LI_transition_threshold_power_HubbardNF17(n_e_avg, B0, A_p, confinement_threshold_scalar=1.0):
    """L-I threshold power, Hubbard NF 2017 scaling (Fig 6 of :cite:`hubbard_threshold_2017`).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # RESOLVED 2026-08-07: Hubbard 2017 fits the LINE-AVERAGED density, so the
    # PROCESS form `li_hubbard2017` is the faithful one and is now the default.
    # This cfspopcon variant substitutes the volume average; kept, gated.
    n19 = n_e_avg / 1.0e19
    return 1.0e6 * (0.162 * (n19 / 10.0) * (B0**0.262) * A_p) * confinement_threshold_scalar


@relation(
    name="L-I transition threshold power AUG",
    tags=("confinement", "tokamak"),
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
    tags=("confinement", "tokamak"),
    outputs="P_LI_thresh",
)
def calc_LI_transition_threshold_power_HubbardNF12(I_p, n_e_avg, confinement_threshold_scalar=1.0):
    """L-I threshold power, Hubbard NF 2012 scaling (Fig 5 of :cite:`hubbard_threshold_2012`).

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # RESOLVED 2026-08-07: as for HubbardNF17 above -- Hubbard's I-mode
    # thresholds are fitted to the LINE-AVERAGED density, so the PROCESS
    # `li_hubbard2012_*` forms are faithful; this one uses the volume average.
    plasma_current = I_p / 1.0e6
    n19 = n_e_avg / 1.0e19
    return 1.0e6 * (2.11 * plasma_current**0.94 * ((n19 / 10.0) ** 0.65)) * confinement_threshold_scalar


@relation(
    name="Ratio of P_SOL to P_LI",
    tags=("confinement", "tokamak"),
    outputs="ratio_of_P_SOL_to_P_LI",
)
def calc_ratio_P_LI(P_sep, P_LI_thresh):
    """Ratio of the power crossing the separatrix to the L-I threshold power.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    """
    # CHECK
    return P_sep / P_LI_thresh


# PROCESS L-H and L-I threshold scalings.
#
# Ported from PROCESS ``process/models/physics/l_h_transition.py``. The
# ``i_l_h_threshold`` enum dispatcher is split into one relation per scaling.
# Every scaling uses the LINE-AVERAGED electron density (PROCESS ``dnla20 =
# nd_plasma_electron_line / 1e20``), mapped to fusdb ``n_la`` -- distinct from
# the volume-averaged density used by fusdb's existing cfspopcon-style L-H
# relation. PROCESS returns thresholds in MW; fusdb stores power in W.
#
# All L-H scalings produce ``P_LH`` and all L-I scalings produce
# ``P_LI_thresh``. Both outputs are gated in variables.yaml, so the existing
# fusdb/cfspopcon defaults stay the defaults unless these relations are
# explicitly included.
#
# The Martin/Snipes ion-mass factor ``(2 / m_ions_total_amu)`` maps
# ``m_ions_total_amu`` -> fusdb ``afuel`` (average ion mass; 2.5 amu for a
# 50/50 D-T mix gives the documented ~20% reduction).

_LH = ("confinement", "tokamak", "process")
_LI = ("confinement", "tokamak", "process")


# --- ITER-1996 (Takizuka) ----------------------------------------------------
@relation(name="L-H threshold ITER-1996 nominal", tags=_LH, outputs="P_LH")
def lh_iter1996_nominal(n_la: float, B0: float, R: float) -> float:
    """ITER-1996 nominal L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.45 * dnla20**0.75 * B0 * R**2)


@relation(name="L-H threshold ITER-1996 upper", tags=_LH, outputs="P_LH")
def lh_iter1996_upper(n_la: float, B0: float, R: float) -> float:
    """ITER-1996 upper L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.3960502816 * dnla20 * B0 * R**2.5)


@relation(name="L-H threshold ITER-1996 lower", tags=_LH, outputs="P_LH")
def lh_iter1996_lower(n_la: float, B0: float, R: float) -> float:
    """ITER-1996 lower L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.5112987149 * dnla20**0.5 * B0 * R**1.5)


# --- Snipes 1997 -------------------------------------------------------------
@relation(name="L-H threshold Snipes-1997 ITER", tags=_LH, outputs="P_LH")
def lh_snipes1997_iter(n_la: float, B0: float, R: float) -> float:
    """Snipes-1997 ITER L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.65 * dnla20**0.93 * B0**0.86 * R**2.15)


@relation(name="L-H threshold Snipes-1997 kappa", tags=_LH, outputs="P_LH")
def lh_snipes1997_kappa(n_la: float, B0: float, R: float, kappa: float) -> float:
    """Snipes-1997 ITER L-H power threshold with elongation factor.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.42 * dnla20**0.80 * B0**0.90 * R**1.99 * kappa**0.76)


# --- Martin 2008 -------------------------------------------------------------
@relation(name="L-H threshold Martin-2008 nominal", tags=_LH, outputs="P_LH")
def lh_martin08_nominal(n_la: float, B0: float, A_p: float, afuel: float) -> float:
    """Martin-2008 nominal L-H power threshold with ion-mass correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # Martin 2008 on the LINE-AVERAGED density, as PROCESS evaluates it.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.0488 * dnla20**0.717 * B0**0.803 * A_p**0.941 * (2.0 / afuel))


@relation(name="L-H threshold Martin-2008 upper", tags=_LH, outputs="P_LH")
def lh_martin08_upper(n_la: float, B0: float, A_p: float, afuel: float) -> float:
    """Martin-2008 upper L-H power threshold with ion-mass correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.05166240355 * dnla20**0.752 * B0**0.835 * A_p**0.96 * (2.0 / afuel))


@relation(name="L-H threshold Martin-2008 lower", tags=_LH, outputs="P_LH")
def lh_martin08_lower(n_la: float, B0: float, A_p: float, afuel: float) -> float:
    """Martin-2008 lower L-H power threshold with ion-mass correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.04609619059 * dnla20**0.682 * B0**0.771 * A_p**0.922 * (2.0 / afuel))


# --- Snipes 2000 -------------------------------------------------------------
@relation(name="L-H threshold Snipes-2000 nominal", tags=_LH, outputs="P_LH")
def lh_snipes2000_nominal(n_la: float, B0: float, R: float, a: float, afuel: float) -> float:
    """Snipes-2000 nominal L-H power threshold with ion-mass correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (1.42 * dnla20**0.58 * B0**0.82 * R * a**0.81 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 upper", tags=_LH, outputs="P_LH")
def lh_snipes2000_upper(n_la: float, B0: float, R: float, a: float, afuel: float) -> float:
    """Snipes-2000 upper L-H power threshold with ion-mass correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (1.547 * dnla20**0.615 * B0**0.851 * R**1.089 * a**0.876 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 lower", tags=_LH, outputs="P_LH")
def lh_snipes2000_lower(n_la: float, B0: float, R: float, a: float, afuel: float) -> float:
    """Snipes-2000 lower L-H power threshold with ion-mass correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (1.293 * dnla20**0.545 * B0**0.789 * R**0.911 * a**0.744 * (2.0 / afuel))


# --- Snipes 2000 closed divertor --------------------------------------------
@relation(name="L-H threshold Snipes-2000 closed divertor nominal", tags=_LH, outputs="P_LH")
def lh_snipes2000_cd_nominal(n_la: float, B0: float, R: float, afuel: float) -> float:
    """Snipes-2000 closed-divertor nominal L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.8 * dnla20**0.5 * B0**0.53 * R**1.51 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 closed divertor upper", tags=_LH, outputs="P_LH")
def lh_snipes2000_cd_upper(n_la: float, B0: float, R: float, afuel: float) -> float:
    """Snipes-2000 closed-divertor upper L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.867 * dnla20**0.561 * B0**0.588 * R**1.587 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 closed divertor lower", tags=_LH, outputs="P_LH")
def lh_snipes2000_cd_lower(n_la: float, B0: float, R: float, afuel: float) -> float:
    """Snipes-2000 closed-divertor lower L-H power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.733 * dnla20**0.439 * B0**0.472 * R**1.433 * (2.0 / afuel))


# --- Martin 2008 with Takizuka aspect-ratio correction ----------------------
def _martin08_aspect_correction(aspect: Any) -> Any:
    """Takizuka aspect-ratio correction; unity for aspect > 2.7.

    Branches per element so the batched popcon namespace evaluates in one call
    (the same reason ``calc_LH_transition_threshold_power`` does).  This form is
    the default ``P_LH`` producer, so it is reached on every scan.
    """
    aspect = np.asarray(aspect, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = 0.098 * aspect / (1.0 - (2.0 / (1.0 + aspect)) ** 0.5)
    return np.where(aspect <= 2.7, corrected, 1.0)


@relation(name="L-H threshold Martin-2008 aspect nominal", tags=_LH, outputs="P_LH")
def lh_martin08_aspect_nominal(n_la: float, B0: float, A_p: float, afuel: float, A: float) -> float:
    """Martin-2008 nominal L-H threshold with Takizuka aspect-ratio correction.

    fusdb's default ``P_LH`` producer, matching PROCESS's own default
    (``i_l_h_threshold = 19``).  Reduces to :func:`lh_martin08_nominal` wherever
    the aspect correction is unity (``A > 2.7``).

    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (
        0.0488 * dnla20**0.717 * B0**0.803 * A_p**0.941 * (2.0 / afuel)
        * _martin08_aspect_correction(A)
    )


@relation(
    name="L-H threshold Martin-2008 aspect nominal (total ion mass)",
    tags=_LH + ("process",),
    outputs="P_LH",
)
def lh_martin08_aspect_nominal_total_ion_mass(
    n_la: float, B0: float, A_p: float, afuel_total: float, A: float
) -> float:
    """Martin-2008 aspect-corrected L-H threshold on the TOTAL ion mass.

    Same fit as :func:`lh_martin08_aspect_nominal`; the only difference is which
    mass the ``2/A_i`` isotope factor uses.  PROCESS evaluates its L-H thresholds
    at ``m_ions_total_amu`` -- every ion, impurities and helium ash included --
    while its confinement scalings use ``m_fuel_amu``.  fusdb's ``afuel`` is the
    fuel mass, so feeding it here overstates the threshold by the mass ratio:
    2/2.514 vs 2/2.731 is 9.2% at the large-tokamak design point, and the factor
    enters linearly.

    Gated; fusdb's fuel-mass form stays the default.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    # CHECK
    return lh_martin08_aspect_nominal.func(
        n_la=n_la, B0=B0, A_p=A_p, afuel=afuel_total, A=A
    )


@relation(name="L-H threshold Martin-2008 aspect upper", tags=_LH, outputs="P_LH")
def lh_martin08_aspect_upper(n_la: float, B0: float, A_p: float, afuel: float, A: float) -> float:
    """Martin-2008 upper L-H threshold with Takizuka aspect-ratio correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    # TODO: Near copy of `lh_martin08_upper` when the aspect correction is
    # unity; check Takizuka/Martin sources to decide which variant to keep and
    # whether the original density is n_la or n_avg.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (
        0.05166240355 * dnla20**0.752 * B0**0.835 * A_p**0.96 * (2.0 / afuel)
        * _martin08_aspect_correction(A)
    )


@relation(name="L-H threshold Martin-2008 aspect lower", tags=_LH, outputs="P_LH")
def lh_martin08_aspect_lower(n_la: float, B0: float, A_p: float, afuel: float, A: float) -> float:
    """Martin-2008 lower L-H threshold with Takizuka aspect-ratio correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    # TODO: Near copy of `lh_martin08_lower` when the aspect correction is
    # unity; check Takizuka/Martin sources to decide which variant to keep and
    # whether the original density is n_la or n_avg.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (
        0.04609619059 * dnla20**0.682 * B0**0.771 * A_p**0.922 * (2.0 / afuel)
        * _martin08_aspect_correction(A)
    )


# --- Hubbard 2012 / 2017 (L -> I-mode thresholds) ---------------------------
@relation(name="L-I threshold Hubbard-2012 nominal", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2012_nominal(I_p: float, n_la: float) -> float:
    """Hubbard-2012 nominal L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # RESOLVED 2026-08-07: Hubbard's I-mode thresholds use the LINE-AVERAGED
    # density (see `li_hubbard2017`), so this n_la form is the faithful one.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (2.11 * (I_p / 1.0e6) ** 0.94 * dnla20**0.65)


@relation(name="L-I threshold Hubbard-2012 upper", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2012_upper(I_p: float, n_la: float) -> float:
    """Hubbard-2012 upper L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    # TODO: Near copy of the Hubbard-2012 nominal/cfspopcon family; check
    # Hubbard 2012 to decide which variants to keep and whether the original
    # density is n_la or n_avg.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (2.11 * (I_p / 1.0e6) ** 1.18 * dnla20**0.83)


@relation(name="L-I threshold Hubbard-2012 lower", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2012_lower(I_p: float, n_la: float) -> float:
    """Hubbard-2012 lower L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    # TODO: Near copy of the Hubbard-2012 nominal/cfspopcon family; check
    # Hubbard 2012 to decide which variants to keep and whether the original
    # density is n_la or n_avg.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (2.11 * (I_p / 1.0e6) ** 0.7 * dnla20**0.47)


@relation(name="L-I threshold Hubbard-2017", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2017(n_la: float, A_p: float, B0: float) -> float:
    """Hubbard-2017 L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # RESOLVED 2026-08-07: Hubbard et al. NF 57 126039 (2017) fits
    # P(L-I)/(n_e S) ~ B_T^0.26 with n_e the LINE-AVERAGED density, so this
    # PROCESS form is the faithful one and is the default; the cfspopcon
    # variant substitutes the volume average and is kept gated.
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.162 * dnla20 * A_p * B0**0.26)
