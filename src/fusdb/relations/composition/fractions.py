"""Species fraction relations."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry import (
    SPECIES,
)

from ._helpers import _positive_denominator, _species_fraction

@relation(
    name="Integrated D fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_D",
)
def integrated_deuterium_fraction_from_density_profiles(n_D: Any, n_fuel: Any) -> Any:
    """Return pointwise deuterium fraction from density profiles."""
    return _species_fraction(n_D, n_fuel, name="f_D")


@relation(
    name="Integrated T fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_T",
)
def integrated_tritium_fraction_from_density_profiles(n_T: Any, n_fuel: Any) -> Any:
    """Return pointwise tritium fraction from density profiles."""
    return _species_fraction(n_T, n_fuel, name="f_T")


@relation(
    name="Integrated He3 fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_He3",
)
def integrated_helium3_fraction_from_density_profiles(n_He3: Any, n_fuel: Any) -> Any:
    """Return pointwise helium-3 fraction from density profiles."""
    return _species_fraction(n_He3, n_fuel, name="f_He3")


@relation(
    name="Integrated He4 fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_He4",
)
def integrated_helium4_fraction_from_density_profiles(n_He4: Any, n_fuel: Any) -> Any:
    """Return pointwise helium-4 fraction from density profiles."""
    return _species_fraction(n_He4, n_fuel, name="f_He4")


@relation(
    name="Integrated p fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_p",
)
def integrated_proton_fraction_from_density_profiles(n_p: Any, n_fuel: Any) -> Any:
    """Return pointwise proton fraction from density profiles."""
    return _species_fraction(n_p, n_fuel, name="f_p")


@relation(
    name="Average fuel mass number",
    tags=("plasma", "composition"),
    outputs="afuel",
)
def average_fuel_mass_number(f_D: Any, f_T: Any, f_He3: Any = 0.0) -> Any:
    """Return average fuel mass number from fuel fractions.

    ``f_He3`` defaults to zero so DT cases can compute ``afuel`` from only
    ``f_D`` and ``f_T``.  If ``f_He3`` exists in the solve namespace, it is used.

    Protons (``f_p``) are deliberately EXCLUDED: this is the average *fuel*
    mass, which the confinement scalings consume as the isotope-effect
    exponent, and protons are ash here (fusdb models no p-burning channel).
    Including them would lighten the apparent fuel mix on a D-He3 point where
    the proton inventory is large.  PROCESS's ``m_fuel_amu`` is fuel-only for
    the same reason.

    Weighted by ``isotopic_mass_u`` (D 2.014, T 3.016), NOT by the mass number
    ``atomic_mass`` (D 2, T 3).  This is a real mass in amu despite the relation
    name: a 50:50 DT mix gives 2.515, not 2.5.  PROCESS's ``m_fuel_amu`` is
    2.51453 on the same mix, so the mass numbers ran a systematic -0.59% low.

    cfspopcon takes the other convention: its ``average_ion_mass`` is a user
    input, supplied as the mass NUMBER 2.5 in the SPARC PRD.  The SPARC fixture
    still derives ``afuel`` here rather than supplying cfspopcon's 2.5, so that
    comparison carries a 0.6% offset on the fuel mass -- ~0.1% on ``tau_E``
    through the isotope exponent, well inside its 6% anchor tolerance.
    """
    fuel_total = _positive_denominator(f_D + f_T + f_He3, name="fuel ion inventory")
    numerator = (
        f_D * float(SPECIES["D"].isotopic_mass_u)
        + f_T * float(SPECIES["T"].isotopic_mass_u)
        + f_He3 * float(SPECIES["He3"].isotopic_mass_u)
    )
    return numerator / fuel_total


@relation(
    name="Average mass of all plasma ions",
    tags=("plasma", "composition"),
    outputs="afuel_total",
)
def average_total_ion_mass_number(
    n_fuel_avg: Any, n_e_avg: Any, n_i_avg: Any,
    f_D: Any, f_T: Any, f_He3: Any = 0.0, f_He4: Any = 0.0, f_p: Any = 0.0,
    c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0, c_O: Any = 0.0,
    c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """Number-density-weighted mean mass of EVERY plasma ion [amu].

    Distinct from :func:`average_fuel_mass_number`: that one is the FUEL mass
    (``afuel``), which the confinement scalings consume as the isotope-effect
    exponent and which therefore excludes protons.  This one is the mass average
    over the whole ion inventory -- fuel, helium ash and impurities included --
    which is what the L-H threshold scalings want.  The two differ by ~9% at
    reactor conditions, and the L-H threshold carries the factor linearly.

    Ported from PROCESS ``physics.py`` (``m_ions_total_amu``), which forms

        [ m_fuel*n_fuel + M_alpha*n_alpha + M_proton*n_proton + m_beam*n_beam
          + sum_{Z>2} n_e*f_imp*m_imp ] / n_ions_total

    Its first three terms are exactly fusdb's ``n_fuel`` bucket (D, T, He3, He4
    and protons, whose fractions ``f_*`` are relative to ``n_fuel``), and its
    ``Z > 2`` sum is exactly fusdb's ``n_imp = n_e * sum_{Li..W} c_X`` -- helium
    is in the fuel bucket here, not in ``c_He``, so counting ``c_He`` as well
    would double it.  fusdb has no fast-ion inventory, so the beam term has no
    counterpart; on a beam-heated design point this runs slightly light.

    Weighted by ``isotopic_mass_u``, not by the mass number -- see
    :func:`average_fuel_mass_number`.

    Adapted from PROCESS; see README.md section "Third-party Notices".
    """
    fuel_masses = {"D": f_D, "T": f_T, "He3": f_He3, "He4": f_He4, "p": f_p}
    impurities = {"Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                  "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    fuel_mass = sum(frac * float(SPECIES[name].isotopic_mass_u) for name, frac in fuel_masses.items())
    impurity_mass = sum(conc * float(SPECIES[name].isotopic_mass_u) for name, conc in impurities.items())
    total_ions = _positive_denominator(n_i_avg, name="total ion inventory")
    return (n_fuel_avg * fuel_mass + n_e_avg * impurity_mass) / total_ions
