"""Impurity composition and dilution relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


from ..utils import _positive_denominator, _species_fraction

# ── Impurity-concentration composition (cfspopcon structure, Mavrin charge states) ──
# cfspopcon's calc_zeff_and_dilution_due_to_impurities builds Z_eff and the ion
# inventory from the supplied c_z = n_z/n_e concentrations and a T-dependent
# mean charge state per species (radas/ADAS interpolators there); these
# relations reproduce that structure with the Mavrin 2018 coronal mean-charge
# fits (the ``zc`` blocks of the mavrin_coronal datasets), so no external
# atomic data is needed.  Evaluated at the volume-averaged temperature, like
# cfspopcon (which feeds average_electron_temp to the interpolator).
#
# The ``Zbar_X`` parameters below are KEYWORD-ONLY WITHOUT DEFAULTS on purpose,
# i.e. real relation inputs rather than constants.  They used to carry the bare
# atomic number as a signature default (``Zbar_Xe: Any = 54.0``), which is not a
# default VALUE but a silent substitution of a different physical model -- fully
# stripped instead of the coronal Zbar(T_e).  A constant is not an edge in the
# dependency graph, so nothing forced the Mavrin producer to run: the relation
# merely used its result if some other path happened to compute it, and fell
# back to the bare charge without an error otherwise.  That is exactly how a
# misfiring gate once turned Zbar_Xe 48.7 into 54 and c_Xe -23% with every
# reactor still reporting success.  As inputs they are honest edges: each has a
# ``default``-tagged Mavrin producer keyed only on T_e_avg, so they resolve on
# every reactor that matches a device group, and a device that knows its own
# charge states still wins by supplying Zbar_X.  Where they do NOT resolve the
# relation now PRUNES instead of silently substituting a bare charge -- measured
# on Eos, whose only tag is the `steallarator` typo so it matches no device
# group and every Zbar_X is inactive: active relations 61 -> 60.  That pruning is
# the intended behaviour, not a regression.  The bare nuclear charge lives in
# species.yaml as ``atomic_number``; it is not this quantity.
# Do not restore the defaults.
#
# EXCEPTION: ``Mean ion charge from composition`` (chi_e, quasineutrality.py)
# deliberately STILL carries these bare-charge defaults.  Making its Zbar_X real
# inputs routes chi_e through T_e_avg, which reorders the composition solve: 3
# failures in tests/PROCESS_large_tokamak and STELLARIS nfev 2 -> 360.  chi_e
# feeds n_i on every reactor, so it is the one place the extra edge changes the
# solve; it needs its own stage and is not folded into this change.

def _mavrin_charge_terms(T_e_avg: Any, concentrations: dict[str, Any]) -> Any:
    """Yield (concentration, Zbar(T_e_avg)) for each supplied species."""
    from ..radiation.impurity_radiation import mavrin_mean_charge

    for symbol, concentration in concentrations.items():
        # Concentrations may be batched (N, 1) columns in the popcon namespace.
        if np.all(np.asarray(concentration, dtype=float) == 0.0):
            continue
        yield concentration, mavrin_mean_charge(symbol, T_e_avg)


@relation(
    name="Effective charge from impurity concentrations (Mavrin)",
    tags=("plasma", "composition"),
    outputs="Z_eff",
)
def effective_charge_from_impurity_concentrations(
    c_Xe: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_W: Any = 0.0,
    *,
    Zbar_He: Any, Zbar_Li: Any, Zbar_Be: Any, Zbar_C: Any,
    Zbar_N: Any, Zbar_O: Any, Zbar_Ne: Any, Zbar_Ar: Any,
    Zbar_Kr: Any, Zbar_Xe: Any, Zbar_W: Any,
) -> Any:
    """Z_eff = 1 + sum_z c_z * Zbar_z(T_e) * (Zbar_z(T_e) - 1), at T_e_avg.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    ``c_Xe`` has NO DEFAULT while the other ten concentrations do.  That is what
    makes xenon the designated free impurity: a parameter without a default is
    classified as a solver VARIABLE (``relation.py`` ``from_function``), so this
    equation can be inverted for it, while the others are CONSTANTS that a
    reactor may supply but the solver never solves for.  Parameters are matched
    by NAME, never by position -- ``c_Xe`` sits first only because Python
    requires no-default parameters to precede defaulted ones.
    A reactor declaring ``Z_eff`` but no composition therefore gets a xenon
    concentration derived FROM that Z_eff instead of a Z_eff crushed to 1.

    Xenon is the choice because it is the only single species that nearly
    satisfies a declared Z_eff and a declared P_rad at once.  Measured on
    STEP_2024 (Z_eff 2.31, P_rad 297 MW), the concentration each species needs
    to explain Z_eff implies an impurity radiation of: C 0.28x, Ne 0.33x,
    Ar 0.54x, Kr 0.71x, **Xe 1.44x**, W 4.89x of what is required -- and a fuel
    dilution of 26.2%, 14.6%, 7.7%, 4.0%, **2.9%**, 2.5%.  Light species
    over-dilute and under-radiate; tungsten radiates ~5x too much.  Helium
    cannot carry a reactor Z_eff at all (it would need ~65%), and it is ash
    anyway -- it stays gated on ``tau_p`` and comes from the ash balance.
    The starting Z_eff = 1 is the pure-hydrogenic plasma; each impurity adds
    c_z * Zbar * (Zbar - 1) (cfspopcon's ``calc_change_in_zeff``), with the
    mean charge from the Mavrin 2018 coronal fits instead of radas.
    """
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    charges = {"He": Zbar_He, "Li": Zbar_Li, "Be": Zbar_Be, "C": Zbar_C, "N": Zbar_N,
               "O": Zbar_O, "Ne": Zbar_Ne, "Ar": Zbar_Ar, "Kr": Zbar_Kr, "Xe": Zbar_Xe,
               "W": Zbar_W}
    z_effective = 1.0
    for symbol, concentration in concentrations.items():
        zbar = charges[symbol]
        z_effective = z_effective + concentration * zbar * (zbar - 1.0)
    return z_effective


@relation(
    name="Effective charge from all species",
    tags=("plasma", "composition"),
    outputs="Z_eff",
)
def effective_charge_from_all_species(
    c_H: Any,
    c_Xe: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_W: Any = 0.0,
    *,
    Zbar_He: Any, Zbar_Li: Any, Zbar_Be: Any, Zbar_C: Any,
    Zbar_N: Any, Zbar_O: Any, Zbar_Ne: Any, Zbar_Ar: Any,
    Zbar_Kr: Any, Zbar_Xe: Any, Zbar_W: Any,
) -> Any:
    """Z_eff = sum_X c_X Zbar_X(T_e)^2, over EVERY species.

    The single definition.  Each term is a concentration relative to the
    electrons (c_X = n_X/n_e), so the fusion ions and the impurities enter on
    exactly the same footing -- the hydrogenic bundle through ``c_H`` at
    Zbar = 1 (fully stripped at core temperatures), helium and the seeded
    impurities through their Mavrin coronal mean charge at ``T_e_avg``.

    It replaces two partial statements that were whitelisted TOGETHER and so
    enforced simultaneously: ``Effective charge from ion fractions`` (fuel only,
    impurities absent) and ``Effective charge from impurity concentrations
    (Mavrin)`` (impurities present but the fuel assumed purely Z = 1, so helium
    ash was missing from the fuel side).

    ``Zbar`` is a per-species mean charge state, NOT a second effective charge:
    it stays a Mavrin quantity and is the only place the fits are consulted.

    ``c_Xe`` has no default, so it remains the invertible knob -- a reactor that
    declares Z_eff without a composition still gets its xenon from this
    equation.
    """
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    charges = {"He": Zbar_He, "Li": Zbar_Li, "Be": Zbar_Be, "C": Zbar_C, "N": Zbar_N,
               "O": Zbar_O, "Ne": Zbar_Ne, "Ar": Zbar_Ar, "Kr": Zbar_Kr, "Xe": Zbar_Xe,
               "W": Zbar_W}
    z_effective = c_H  # hydrogenic bundle: Zbar = 1, so it enters as c_H itself
    for symbol, concentration in concentrations.items():
        if np.all(np.asarray(concentration, dtype=float) == 0.0):
            continue
        zbar = charges[symbol]
        z_effective = z_effective + concentration * zbar * zbar
    return z_effective


@relation(
    name="Fuel dilution from impurity concentrations (cfspopcon)",
    tags=("plasma", "composition"),
    outputs="dilution",
)
def dilution_from_impurity_concentrations(
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
    *,
    Zbar_He: Any, Zbar_Li: Any, Zbar_Be: Any, Zbar_C: Any,
    Zbar_N: Any, Zbar_O: Any, Zbar_Ne: Any, Zbar_Ar: Any,
    Zbar_Kr: Any, Zbar_Xe: Any, Zbar_W: Any,
) -> Any:
    """Fuel dilution, cfspopcon's SIMPLIFIED form: 1 - sum_X c_X Zbar_X(T_e).

    TERMINOLOGY (project-wide): ``f_X`` are FRACTIONS (n_X/n_i), ``c_X`` are
    CONCENTRATIONS (n_X/n_e), and FUEL DILUTION is the fuel share of the
    electron density.

    cfspopcon reaches the fuel dilution from the impurity side and counts only
    D and T as fuel.  That is EXACT under its own assumptions -- it has no He3
    fuel and treats all helium as an impurity -- so this relation reproduces
    cfspopcon faithfully and is what tests/cfspopcon_SPARC selects by name.
    Here it is a SIMPLIFICATION: what it actually computes is the HYDROGENIC
    concentration (p + D + T)/n_e, identically ``c_H``, which drops He3 (fuel
    on a D-He3 machine) and keeps protium (ash).  ``Fuel dilution from fuel
    fractions`` is the general form and the default.  The sum runs over every
    NON-HYDROGENIC element (helium included), so what is left is protium +
    deuterium + tritium:

        c_H + sum_{X != H} c_X Zbar_X = 1   (charge neutrality, hydrogen at Z=1)
        =>  1 - sum_{X != H} c_X Zbar_X  ==  c_H,  identically

    so this relation and ``Hydrogenic concentration from fuel fractions``
    compute the SAME quantity by two routes.  They agree because the system is
    consistent, not because anything asserts it -- see variables.yaml.

    The FUEL DILUTION is a DIFFERENT number -- (n_D + n_T + 2 n_He3)/n_e on the
    charge-weighted convention: it drops protium (ash) and adds He3 (fuel on a
    D-He3 machine, where He3 sits in the c_He term above at Zbar = 2 and is
    therefore EXCLUDED here).  The two coincide only when there is no protium
    and no He3 -- true of every D-T case in the test suite, which is why the
    distinction is easy to miss.  **Nothing currently computes the fuel
    dilution**; it needs the isotope split, so it cannot be recovered from the
    element concentrations alone.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    cfspopcon calls this ``dilution`` and glosses it as n_DT/n_e, which is
    consistent THERE because cfspopcon has no He3 fuel and counts all helium as
    an impurity -- so its fuel and hydrogenic fractions are the same thing.
    Mean charge from the Mavrin 2018 coronal fits instead of radas.
    """
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    charges = {"He": Zbar_He, "Li": Zbar_Li, "Be": Zbar_Be, "C": Zbar_C, "N": Zbar_N,
               "O": Zbar_O, "Ne": Zbar_Ne, "Ar": Zbar_Ar, "Kr": Zbar_Kr, "Xe": Zbar_Xe,
               "W": Zbar_W}
    dilution_terms = 0.0
    for symbol, concentration in concentrations.items():
        dilution_terms = dilution_terms + concentration * charges[symbol]
    return np.maximum(1.0 - dilution_terms, 0.0)


@relation(
    name="Ion density average from impurity concentrations (Mavrin)",
    tags=("plasma", "composition"),
    outputs="n_i_avg",
)
def ion_density_average_from_impurity_concentrations(
    n_e_avg: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
    *,
    Zbar_He: Any, Zbar_Li: Any, Zbar_Be: Any, Zbar_C: Any,
    Zbar_N: Any, Zbar_O: Any, Zbar_Ne: Any, Zbar_Ar: Any,
    Zbar_Kr: Any, Zbar_Xe: Any, Zbar_W: Any,
) -> Any:
    """n_i_avg = n_e_avg * (1 - sum_z c_z * (Zbar_z(T_e) - 1)), at T_e_avg.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    cfspopcon's dilution is n_fuel/n_e = 1 - sum(c_z * Zbar_z); this adds the
    impurity ions back, n_i = n_fuel + n_e * sum(c_z).
    Mean charge from the Mavrin 2018 coronal fits instead of radas.

    INCONSISTENT with the pinned definition of ``n_i_avg`` (fusion ions only --
    p, D, T, He3, He4 -- see variables.yaml): the ``+ n_e * sum(c_z)`` counts
    the impurity ions, which that definition excludes.  It also states only
    HALF the charge balance -- the impurity term with no fuel ``zbar`` -- while
    ``Ion density average from quasineutrality`` states the other half, the
    fuel ``zbar`` with no impurity term.  Neither is complete; see
    .claude/scratchpad.md.
    """
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    charges = {"He": Zbar_He, "Li": Zbar_Li, "Be": Zbar_Be, "C": Zbar_C, "N": Zbar_N,
               "O": Zbar_O, "Ne": Zbar_Ne, "Ar": Zbar_Ar, "Kr": Zbar_Kr, "Xe": Zbar_Xe,
               "W": Zbar_W}
    dilution_terms = 0.0
    for symbol, concentration in concentrations.items():
        dilution_terms = dilution_terms + concentration * (charges[symbol] - 1.0)
    return n_e_avg * np.maximum(1.0 - dilution_terms, 0.0)


# ── isotope-fraction -> element-concentration bridges ────────────────────────
# fusdb carries composition two ways, and both are load-bearing:
#   f_X  isotope-keyed, denominated in n_i -- what the NUCLEAR side needs
#        (reactivity distinguishes D from T from He3; the ash balance and
#        quasineutrality are written per isotope).
#   c_X  element-keyed, denominated in n_e -- what the ATOMIC side needs
#        (Lz cooling curves, Mavrin Zbar, Z_eff, NBI stopping are set by
#        electronic structure, so isotopes are indistinguishable).
# The bridge is therefore one relation per ELEMENT that has tracked isotope
# fractions: c_X = (sum of that element's isotope fractions) * n_i/n_e.
# Elements with no isotope resolution (C, Ne, Xe, W, ...) are declared as c_X
# directly and need no bridge.  A reactor may state both sides; reconcile then
# treats the bridge as any other constraint and resolves the disagreement.


def _element_concentration(fractions: tuple, n_i_avg: Any, n_e_avg: Any) -> Any:
    """c_X = (sum of the element's isotope fractions) * n_i/n_e."""
    total = fractions[0]
    for item in fractions[1:]:
        total = total + item
    return total * n_i_avg / _positive_denominator(n_e_avg, name="n_e_avg")


@relation(
    name="Helium concentration from ash densities",
    tags=("plasma", "composition"),
    outputs="c_He",
)
def helium_concentration_from_ash_densities(
    f_He4: Any,
    n_i_avg: Any,
    n_e_avg: Any,
    f_He3: Any = 0.0,
) -> Any:
    """c_He = (f_He3 + f_He4) * n_i/n_e -- the He bridge.

    Helium is carried isotope-resolved (f_He3/f_He4, which the ash balance
    needs) and, for the atomic data, as the per-element concentration
    ``c_He = n_He/n_e``.
    """
    return _element_concentration((f_He3, f_He4), n_i_avg, n_e_avg)


@relation(
    name="Hydrogenic concentration from fuel fractions",
    tags=("plasma", "composition"),
    outputs="c_H",
)
def hydrogen_concentration_from_fuel_fractions(
    f_D: Any,
    f_T: Any,
    f_p: Any,
    n_i_avg: Any,
    n_e_avg: Any,
) -> Any:
    """c_H = (f_D + f_T + f_p) * n_i/n_e -- the hydrogen bridge.

    ``f_p`` is the PROTON (ionised protium) fraction, and protons are a fusion
    product as much as a fuel: ``DDp``, ``DHe3``, ``He3He3`` and ``THe3_np`` all
    yield them (reactions.yaml).  On a D-He3 machine they are a primary ash
    channel comparable to He4, and being Z=1 they radiate on hydrogen's cooling
    curve and dilute like any other ion.
    """
    return _element_concentration((f_D, f_T, f_p), n_i_avg, n_e_avg)
