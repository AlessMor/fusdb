"""Build a fusdb reactor fixture from any PROCESS MFILE.

One PROCESS run in, one fusdb ``Reactor`` out. Used by every
``test_PROCESS_large_tokamak_*.py`` module and by the notebooks, so the mapping
from PROCESS to fusdb lives in exactly one place.

What is SUPPLIED to fusdb (PROCESS's independent design vector) and what is left
FREE (so fusdb derives it and the comparison is a real test) is documented
inline below -- that split is the whole substance of the comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _process_mfile import read_mfile

HERE = Path(__file__).parent
REFERENCE = HERE / "reference"

# fusdb's registry default is 46, which is NOT enough here.  P_aux is a small
# difference of two large numbers (heating power minus alpha power), so it
# inherits the quadrature error of the profile integrals.  Measured convergence
# against PROCESS at this design point:
#
#     grid   P_fus     P_aux     Q
#       46   +6.40%   -3.66%   +11.26%
#      101   +6.43%   -0.66%    +7.94%
#      201   +6.44%   -1.25%    +8.58%
#      401   +6.44%   -1.52%    +8.88%
#
# P_fus was converged by 46 points, so its +6.4% could not be quadrature -- but
# it was NOT a model difference either.  RESOLVED 2026-08-03: fusdb's ion
# inventory omitted the impurity electrons, so it burned fuel the impurities
# should have displaced.  With `chi_e = n_e/n_i` carrying the full charge
# balance (see relations/composition/quasineutrality.py) the gap closes to
# **+0.27%**.  Q only comes inside 10% once the grid is refined.
PROFILE_SIZE = 201

# PROCESS impurity array, 0-BASED as written to the MFILE (the input file's
# f_nd_impurity_electrons(N) is 1-based, so IN.DAT (13) == MFILE index 12).
IMP_LABELS = ["H", "He", "Be", "C", "N", "O", "Ne", "Si", "Ar", "Fe", "Ni", "Kr", "Xe", "W"]


# --------------------------------------------------------------------------
# profiles: PROCESS's own analytic pedestal form
# --------------------------------------------------------------------------
# Transcribed verbatim from process/models/physics/profiles.py
# (NProfile/TProfile.calculate_profile_y). PROCESS runs i_plasma_pedestal = 1,
# an H-mode pedestal (HELIOS parameterisation) that fusdb's parabolic-via-
# peaking family cannot represent. Rather than approximate it with a peaking
# factor, we rebuild PROCESS's exact profile and supply it: a supplied profile
# makes fusdb's generator relations inactive and is authoritative.


def n_profile(rho, n0, n_ped, n_sep, rho_ped, alphan):
    y = np.empty_like(rho)
    core = rho <= rho_ped
    y[core] = n_ped + (n0 - n_ped) * (1 - (rho[core] / rho_ped) ** 2) ** alphan
    y[~core] = n_sep + (n_ped - n_sep) * (1 - rho[~core]) / (1 - rho_ped)
    return y


def t_profile(rho, t0, t_ped, t_sep, rho_ped, alphat, tbeta):
    y = np.empty_like(rho)
    core = rho <= rho_ped
    y[core] = t_ped + (t0 - t_ped) * (1 - (rho[core] / rho_ped) ** tbeta) ** alphat
    y[~core] = t_sep + (t_ped - t_sep) * (1 - rho[~core]) / (1 - rho_ped)
    return y


def _get(data, key, default=None):
    entry = data.get(key)
    if entry is None:
        if default is None:
            raise KeyError(f"MFILE has no key {key!r}")
        return default
    return entry[1]


def impurity_fractions(data):
    """Recover n_X/n_e for each impurity from the MFILE density profiles.

    The MFILE writes impurity *densities* per profile point
    (``f_nd_impurity_electrons<idx>_<i>``), so the concentration is the on-axis
    density over the on-axis electron density. Reading it this way works for
    both evaluation and optimisation runs -- in the latter the Xe fraction is an
    iteration variable whose converged value differs from the input file's
    initial guess (3.8e-4 -> 5.97e-4 in the introduction run), and the input
    file would give the wrong composition.
    """
    n_e0 = _get(data, "nd_plasma_electron_on_axis")
    out = {}
    for index, label in enumerate(IMP_LABELS):
        key = f"f_nd_impurity_electrons{index}_0"
        if key in data:
            out[label] = data[key][1] / n_e0
    return out


def build_config(mfile_path, profile_size=PROFILE_SIZE):
    """Return a fusdb reactor config dict mirroring one PROCESS run."""
    data = read_mfile(mfile_path)
    rho = np.linspace(0.0, 1.0, profile_size)

    n_e = n_profile(
        rho,
        _get(data, "nd_plasma_electron_on_axis"),
        _get(data, "nd_plasma_pedestal_electron"),
        _get(data, "nd_plasma_separatrix_electron"),
        _get(data, "radius_plasma_pedestal_density_norm"),
        _get(data, "alphan"),
    )
    t_e = t_profile(
        rho,
        _get(data, "temp_plasma_electron_on_axis_kev"),
        _get(data, "temp_plasma_pedestal_kev"),
        _get(data, "temp_plasma_separatrix_kev"),
        _get(data, "radius_plasma_pedestal_temp_norm"),
        _get(data, "alphat"),
        _get(data, "tbeta"),
    )

    impurities = impurity_fractions(data)

    # Convert PROCESS's composition onto fusdb's total-ion denominator.
    _n_e = _get(data, "nd_plasma_electrons_vol_avg")
    _n_fuel = _get(data, "nd_plasma_fuel_ions_vol_avg")
    _n_he = _get(data, "f_nd_alpha_thermal_electron") * _n_e
    _n_imp = sum(
        fraction * _n_e
        for label, fraction in impurities.items()
        if label not in ("H", "He")
    )
    _n_i_total = _n_fuel + _n_he + _n_imp

    variables = {
        # --- geometry -----------------------------------------------------
        "R": {"value": _get(data, "rmajor"), "fixed": True},
        "a": {"value": _get(data, "rminor"), "fixed": True},
        # ELONGATION -- four distinct quantities on BOTH sides, so each is
        # supplied to its own variable rather than one value being reused:
        #   separatrix        1.85000  PROCESS's own `kappa` input
        #   areal             1.71879  S_phi/(pi a^2)  -> fusdb's `kappa_areal`
        #   kappa_ipb         1.68145  volume-equivalent, V_p/(2 pi^2 R a^2)
        #   kappa_95          1.65179  95% surface, what current/L-H fits use
        # PROCESS does not report an areal elongation directly, so it is recovered
        # from its cross-section -- which is the definition of kappa_areal.
        #
        # `kappa` is PROCESS's own `kappa` input -- the SEPARATRIX elongation, which
        # is what fusdb's `kappa` means (kappa == kappa_sep == kappa_geom at
        # psi_N = 1).  Nine PROCESS-imported relations here read it expecting exactly
        # that: Snipes-1997, Stambaugh beta, Menard internal inductance, the FIESTA
        # and Peng current coefficients, Albajar-Fidone synchrotron and Wong
        # bootstrap.  It used to be pinned to the AREAL value, so all nine were
        # silently fed the wrong elongation.
        "kappa": {"value": _get(data, "kappa."), "fixed": True},
        "kappa_areal": {
            "value": _get(data, "a_plasma_poloidal") / (np.pi * _get(data, "rminor") ** 2),
            "fixed": True,
        },
        "kappa_95": {"value": _get(data, "kappa95"), "fixed": True},
        # V_p is recovered EXACTLY from this by letting "IPB elongation from
        # volume" run backwards -- it is how PROCESS defines kappa_ipb.
        "kappa_ipb": {"value": _get(data, "kappa_ipb"), "fixed": True},
        "separatrix_elongation": {"value": _get(data, "kappa."), "fixed": True},
        "delta_95": {"value": _get(data, "triang95"), "fixed": True},
        # i_plasma_geometry = 0 is a pure kappa/triang shape, no squareness.
        # Fixed, not merely supplied: left free the solver walks it far enough
        # to collapse V_p and S_phi.
        "squareness": {"value": 0.0, "fixed": True},
        # --- field and current --------------------------------------------
        "B0": {"value": _get(data, "b_plasma_toroidal_on_axis"), "fixed": True},
        # I_p is SUPPLIED and qstar left FREE, deliberately. fusdb's f_shaping
        # and PROCESS's IPDG89 are the same formula fed different elongations
        # (areal vs kappa_95), a ~19% difference. Pinning qstar would force
        # that contradiction into the solve and poison everything downstream;
        # supplying I_p instead lets the comparison REPORT the difference via
        # qstar. See FINDINGS.md finding 1.
        "I_p": {"value": _get(data, "plasma_current"), "fixed": True},
        # fusdb's q95 has no producer; it is a pure input feeding diagnostics.
        "q95": {"value": _get(data, "q95"), "fixed": True},
        # --- confinement ---------------------------------------------------
        # PROCESS's `hfact` aliases to fusdb's H_factor -- the GENERIC multiplier
        # applied to whichever scaling is active, which is exactly what hfact is
        # (PROCESS applies it to whatever i_confinement_time selects).  NOT
        # H98_y2: that is the scaling-specific factor, and fusdb keeps the two
        # separate so they compose multiplicatively.
        "H_factor": {"value": _get(data, "hfact"), "fixed": True},
        # --- profiles (PROCESS's own, supplied directly) --------------------
        # n_e_avg / T_e_avg / T_i_avg are NOT supplied: a fixed profile pins its
        # own scalar average by construction and supplying both double-pins it.
        # T_i == T_e at these design points (MFILE
        # temp_plasma_ion_vol_avg_kev == temp_plasma_electron_vol_avg_kev).
        "n_e": {"value": [float(v) for v in n_e], "fixed": True},
        "T_e": {"value": [float(v) for v in t_e], "unit": "keV", "fixed": True},
        "T_i": {"value": [float(v) for v in t_e], "unit": "keV", "fixed": True},
        # --- composition ----------------------------------------------------
        # n_i is deliberately NOT supplied: fusdb derives n_i_avg from its own
        # quasineutrality + Mavrin mean charges, which is under test.
        # DENOMINATOR CONVENTION -- easy to get wrong, and costly.  fusdb's
        # f_D / f_T / f_He4 are fractions of the TOTAL ion density
        # (n_D = f_D * n_i), whereas PROCESS quotes the helium fraction against
        # the ELECTRON density and its 50:50 D-T split against the FUEL ions.
        # Setting f_D = f_T = 0.5 directly leaves no room for helium or
        # impurities, over-stating the fuel by ~5% and hence P_fus by ~10%
        # (P_fus ~ n_D n_T ~ n_fuel^2), which then propagates into P_aux and Q.
        "f_He4": {"value": _n_he / _n_i_total, "fixed": True},
        "f_D": {"value": 0.5 * _n_fuel / _n_i_total, "fixed": True},
        "f_T": {"value": 0.5 * _n_fuel / _n_i_total, "fixed": True},
        # --- radiation ------------------------------------------------------
        "f_sync_reflect": {"value": _get(data, "f_sync_reflect"), "fixed": True},
        # PROCESS uses m_fuel_amu in the confinement scalings and
        # m_ions_total_amu in the L-H thresholds; they differ by ~9%.
        "afuel_total": {"value": _get(data, "m_ions_total_amu"), "fixed": True},
        # PROCESS core/edge radiation split parameters, carried verbatim.
        "radius_plasma_core_norm": {
            "value": _get(data, "radius_plasma_core_norm"), "fixed": True},
        "f_p_plasma_core_rad_reduction": {
            "value": _get(data, "f_p_plasma_core_rad_reduction"), "fixed": True},
        # Load-bearing: PROCESS's synchrotron model (Albajar-Fidone) takes the
        # alpha-index parameterisation directly. Without these it is pruned for
        # missing inputs, and so is fusdb's cfspopcon form (which needs
        # separatrix_elongation) -- leaving NO P_sync producer, which prunes
        # "Total radiated power" and "Power crossing the separatrix" with it.
        "alphan": {"value": _get(data, "alphan"), "fixed": True},
        "alphat": {"value": _get(data, "alphat"), "fixed": True},
        "tbeta": {"value": _get(data, "tbeta"), "fixed": True},
    }

    for label, fraction in impurities.items():
        if label in ("H", "He"):
            # PROCESS sets the hydrogenic entry from quasineutrality and the
            # helium entry from f_nd_alpha_thermal_electron (already supplied
            # as f_He4); neither is a free impurity input.
            continue
        if fraction > 0.0:
            variables[f"c_{label}"] = {"value": fraction, "fixed": True}

    return {
        "metadata": {
            "id": "PROCESS_large_tokamak",
            "name": "Generic large tokamak (PROCESS)",
            "organization": "UKAEA / PROCESS",
            "notes": f"Built by _fixture.py from {Path(mfile_path).name}.",
        },
        "tags": ["tokamak", "h_mode"],
        "variables": variables,
        "relations": {
            "include": [
                # i_bootstrap_current = 4 -> Sauter. fusdb defaults to the
                # cfspopcon Gi form; the MFILE confirms the selection
                # (f_c_plasma_bootstrap == f_c_plasma_bootstrap_sauter).
                "Bootstrap fraction Sauter",
                # PROCESS radiates synchrotron with Albajar-Fidone.
                "Synchrotron radiation Albajar-Fidone",
                # PROCESS feeds the confinement scaling a transport loss power
                # with the CORE radiation subtracted (i_rad_loss = 1); fusdb's
                # default subtracts no radiation at all. Without these the
                # comparison is on mismatched definitions and P_loss/tau_E/
                # P_sep/P_aux/Q all sit 40-60% out.
                "Core radiation power (PROCESS)",
                "Plasma heating power (PROCESS)",
                "Plasma loss power (PROCESS)",
                "Power crossing the separatrix (PROCESS)",
                # i_l_h_threshold = 19 -> Martin 2008 ASPECT-RATIO CORRECTED,
                # nominal.  fusdb's default L-H relation runs +23% high here,
                # which pushes P_sep below threshold and demotes the regime.
                # i_l_h_threshold = 19, evaluated on the TOTAL ion mass as
                # PROCESS does (fusdb's default form uses the fuel mass).
                "L-H threshold Martin-2008 aspect nominal (total ion mass)",
                # PROCESS's i_plasma_current = 4 evaluates the IPDG89 shaping fit
                # at kappa_95; fusdb's default evaluates it at the areal kappa.
                "Plasma shaping function for q_star (PROCESS IPDG89)",
                # Mavrin's Lz is the TOTAL impurity cooling rate, bremsstrahlung
                # included.  fusdb's default P_brem is Z_eff-weighted, so it adds
                # the impurity bremsstrahlung a second time: P_rad ran +10.1% and
                # dragged P_sep to -20.9%.  The hydrogenic-only form is the
                # composition-consistent partner for a Mavrin P_line.
            ],
            "exclude": [
                # `include` ADDS rather than replaces, so each selected relation
                # needs its fusdb default dropped or the output is
                # over-determined and lands in failed_relations.
                "Bootstrap current fraction",
                "Synchrotron radiation",
                "Plasma loss power",
                "L-H transition threshold power",
                "Plasma shaping function for q_star",
                # kappa_95 is supplied, so fusdb's kappa->kappa_95 derivation
                # would over-determine it.
                "Elongation 95%",
                "Power crossing the separatrix",
                # Enforces P_in == P_loss (cfspopcon's convention, where P_loss
                # already includes the core radiation).  PROCESS keeps the two
                # distinct -- P_loss = P_in - P_rad_core -- so this identity
                # directly contradicts "Plasma loss power (PROCESS)".
                # V_p: fusdb's Sauter form, fed the IPB elongation, runs -7.2%
                # against PROCESS -- which propagates to W_th, P_loss and P_aux.
                # Dropping it lets "IPB elongation from volume" (kappa_ipb =
                # V_p / (2 pi^2 R a^2)) run backwards from the supplied
                # kappa_ipb and reproduce PROCESS's volume EXACTLY, which is how
                # PROCESS defines kappa_ipb in the first place.
                "Tokamak plasma volume",
                # Same idiom for the cross-section: kappa_areal is supplied from
                # PROCESS's own a_plasma_poloidal, so letting "Areal elongation
                # from cross-section" (kappa_areal = S_phi / (pi a^2)) run
                # backwards reproduces PROCESS's S_phi EXACTLY.  fusdb's Sauter
                # form would instead derive S_phi from `kappa` (now correctly the
                # separatrix 1.85) and over-determine it.
                # Measured 2026-07-31: S_phi and V_p now both land at -0.0000%
                # against PROCESS, where S_phi previously ran -3.2%.
                "Tokamak plasma cross-sectional surface",
            ],
        },
    }


def build_reactor(mfile_path, profile_size=PROFILE_SIZE):
    """Build and return a solved-ready fusdb ``Reactor`` for one PROCESS run."""
    import tempfile

    import yaml

    import fusdb

    config = build_config(mfile_path, profile_size)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return fusdb.Reactor.from_yaml(path)


# --------------------------------------------------------------------------
# the reference runs
# --------------------------------------------------------------------------

INTRODUCTION = REFERENCE / "introduction" / "large_tokamak_MFILE.DAT"
EVAL_POINT = REFERENCE / "eval_point" / "large_tokamak_eval_MFILE.DAT"


def _sweep(directory):
    root = REFERENCE / directory
    return {
        point.name: point / "large_tokamak_eval_MFILE.DAT"
        for point in sorted(root.iterdir())
        if point.is_dir()
    }


def tungsten_sweep():
    """{tag: mfile_path} for the W sensitivity sweep, ordered by W fraction."""
    points = _sweep("tungsten_sweep")
    return dict(sorted(points.items(), key=lambda kv: float(kv[0].split("_")[1])))


def plasma_variants():
    """{tag: mfile_path} for the one-input-at-a-time plasma variants."""
    return _sweep("plasma_variants")
