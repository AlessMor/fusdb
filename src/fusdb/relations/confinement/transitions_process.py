"""L-H and L-I transition power-threshold scalings from PROCESS.

Ported from PROCESS ``process/models/physics/l_h_transition.py``. The
``i_l_h_threshold`` enum dispatcher is split into one relation per scaling.
Every scaling uses the LINE-AVERAGED electron density (PROCESS ``dnla20 =
nd_plasma_electron_line / 1e20``), mapped to fusdb ``n_la`` -- distinct from the
volume-averaged density used by fusdb's existing (cfspopcon) L-H relations.
PROCESS returns thresholds in MW; fusdb stores power in W.

All L-H scalings produce ``P_LH`` and all L-I scalings produce ``P_LI_thresh``,
both gated in variables.yaml so the existing fusdb/cfspopcon defaults stay the
defaults.

The Martin/Snipes ion-mass factor ``(2 / m_ions_total_amu)`` maps
``m_ions_total_amu`` -> fusdb ``afuel`` (average ion mass; 2.5 amu for a 50/50
D-T mix gives the documented ~20% reduction).
"""

from fusdb import relation

_LH = ("confinement", "h_mode", "process")
_LI = ("confinement", "i_mode", "process")


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
    """Martin-2008 nominal L-H power threshold (with ion-mass correction).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.0488 * dnla20**0.717 * B0**0.803 * A_p**0.941 * (2.0 / afuel))


@relation(name="L-H threshold Martin-2008 upper", tags=_LH, outputs="P_LH")
def lh_martin08_upper(n_la: float, B0: float, A_p: float, afuel: float) -> float:
    """Martin-2008 upper L-H power threshold (with ion-mass correction).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.05166240355 * dnla20**0.752 * B0**0.835 * A_p**0.96 * (2.0 / afuel))


@relation(name="L-H threshold Martin-2008 lower", tags=_LH, outputs="P_LH")
def lh_martin08_lower(n_la: float, B0: float, A_p: float, afuel: float) -> float:
    """Martin-2008 lower L-H power threshold (with ion-mass correction).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.04609619059 * dnla20**0.682 * B0**0.771 * A_p**0.922 * (2.0 / afuel))


# --- Snipes 2000 -------------------------------------------------------------
@relation(name="L-H threshold Snipes-2000 nominal", tags=_LH, outputs="P_LH")
def lh_snipes2000_nominal(n_la: float, B0: float, R: float, a: float, afuel: float) -> float:
    """Snipes-2000 nominal L-H power threshold (with ion-mass correction).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (1.42 * dnla20**0.58 * B0**0.82 * R * a**0.81 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 upper", tags=_LH, outputs="P_LH")
def lh_snipes2000_upper(n_la: float, B0: float, R: float, a: float, afuel: float) -> float:
    """Snipes-2000 upper L-H power threshold (with ion-mass correction).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (1.547 * dnla20**0.615 * B0**0.851 * R**1.089 * a**0.876 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 lower", tags=_LH, outputs="P_LH")
def lh_snipes2000_lower(n_la: float, B0: float, R: float, a: float, afuel: float) -> float:
    """Snipes-2000 lower L-H power threshold (with ion-mass correction).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (1.293 * dnla20**0.545 * B0**0.789 * R**0.911 * a**0.744 * (2.0 / afuel))


# --- Snipes 2000 closed divertor ---------------------------------------------
@relation(name="L-H threshold Snipes-2000 closed divertor nominal", tags=_LH, outputs="P_LH")
def lh_snipes2000_cd_nominal(n_la: float, B0: float, R: float, afuel: float) -> float:
    """Snipes-2000 closed-divertor nominal L-H power threshold (ion-mass corrected).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.8 * dnla20**0.5 * B0**0.53 * R**1.51 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 closed divertor upper", tags=_LH, outputs="P_LH")
def lh_snipes2000_cd_upper(n_la: float, B0: float, R: float, afuel: float) -> float:
    """Snipes-2000 closed-divertor upper L-H power threshold (ion-mass corrected).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.867 * dnla20**0.561 * B0**0.588 * R**1.587 * (2.0 / afuel))


@relation(name="L-H threshold Snipes-2000 closed divertor lower", tags=_LH, outputs="P_LH")
def lh_snipes2000_cd_lower(n_la: float, B0: float, R: float, afuel: float) -> float:
    """Snipes-2000 closed-divertor lower L-H power threshold (ion-mass corrected).
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.733 * dnla20**0.439 * B0**0.472 * R**1.433 * (2.0 / afuel))


# --- Martin 2008 with Takizuka aspect-ratio correction -----------------------
def _martin08_aspect_correction(aspect: float) -> float:
    """Takizuka aspect-ratio correction (unity for aspect > 2.7)."""
    if aspect <= 2.7:
        return 0.098 * aspect / (1.0 - (2.0 / (1.0 + aspect)) ** 0.5)
    return 1.0


@relation(name="L-H threshold Martin-2008 aspect nominal", tags=_LH, outputs="P_LH")
def lh_martin08_aspect_nominal(n_la: float, B0: float, A_p: float, afuel: float, A: float) -> float:
    """Martin-2008 nominal L-H threshold with Takizuka aspect-ratio correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (
        0.0488 * dnla20**0.717 * B0**0.803 * A_p**0.941 * (2.0 / afuel)
        * _martin08_aspect_correction(A)
    )


@relation(name="L-H threshold Martin-2008 aspect upper", tags=_LH, outputs="P_LH")
def lh_martin08_aspect_upper(n_la: float, B0: float, A_p: float, afuel: float, A: float) -> float:
    """Martin-2008 upper L-H threshold with Takizuka aspect-ratio correction.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
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
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (
        0.04609619059 * dnla20**0.682 * B0**0.771 * A_p**0.922 * (2.0 / afuel)
        * _martin08_aspect_correction(A)
    )


# --- Hubbard 2012 / 2017 (L -> I-mode thresholds) ----------------------------
@relation(name="L-I threshold Hubbard-2012 nominal", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2012_nominal(I_p: float, n_la: float) -> float:
    """Hubbard-2012 nominal L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (2.11 * (I_p / 1.0e6) ** 0.94 * dnla20**0.65)


@relation(name="L-I threshold Hubbard-2012 upper", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2012_upper(I_p: float, n_la: float) -> float:
    """Hubbard-2012 upper L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (2.11 * (I_p / 1.0e6) ** 1.18 * dnla20**0.83)


@relation(name="L-I threshold Hubbard-2012 lower", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2012_lower(I_p: float, n_la: float) -> float:
    """Hubbard-2012 lower L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (2.11 * (I_p / 1.0e6) ** 0.7 * dnla20**0.47)


@relation(name="L-I threshold Hubbard-2017", tags=_LI, outputs="P_LI_thresh")
def li_hubbard2017(n_la: float, A_p: float, B0: float) -> float:
    """Hubbard-2017 L-I power threshold.
    Adapted from PROCESS; see README.md section "Third-party Notices"."""
    # CHECK
    dnla20 = n_la / 1.0e20
    return 1.0e6 * (0.162 * dnla20 * A_p * B0**0.26)
