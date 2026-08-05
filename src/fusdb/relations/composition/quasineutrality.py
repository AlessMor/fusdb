"""Quasineutrality and composition helper relations."""

from typing import Any

from fusdb.relation import relation
from ..utils import _positive_denominator


@relation(
    name="Ion density average from quasineutrality",
    tags=("plasma", "composition"),
    outputs="n_i_avg",
)
def ion_density_average_from_quasineutrality(
    n_e_avg: Any,
    Zbar_i_avg: Any,
) -> Any:
    """<n_i> = <n_e> / Zbar_i_avg -- quasineutrality at the volume average.

    ``Zbar_i_avg = n_e/n_i = sum_s Z_s f_s`` is the SCALAR mean charge per ion,
    supplied by ``Mean ion charge from composition``.  This average-level anchor
    is where quasineutrality (and hence the dilution) actually binds: it keeps
    the ion inventory physical even when the ion *profile shape* is derived
    independently of the electron one (its own peaking scaling + generator).
    The pointwise tie below is only a weak default, so without this relation a
    decoupled ion side would float free of ``n_e``.

    Reads the composition-side scalar ``Zbar_i_avg`` rather than the local
    density-ratio profile ``chi_e``, so the ion level never depends on a density
    ratio -- that keeps ``c_Xe`` off the ``c_Xe -> n_i_avg -> Z_eff -> c_Xe``
    inversion loop.  It used to inline the FUEL-ONLY ``zbar`` (dropping the
    impurity electrons and pinning n_i/n_e to 1); ``Zbar_i_avg`` is that zbar
    divided by ``1 - sum_z c_z Zbar_z``.
    """
    return n_e_avg / _positive_denominator(Zbar_i_avg, name="mean ion charge")


@relation(
    name="Ion density from electron density (quasineutrality)",
    tags=("default", "plasma", "composition"),
    outputs="n_i",
)
def ion_density_from_electron_density(
    n_e: Any,
    Zbar_i: Any,
) -> Any:
    """n_i(rho) = n_e(rho) / Zbar_i(rho), pointwise -- quasineutrality.

    Quasineutrality is CHARGE balance, not particle balance.  The electrons
    balance the positive CHARGE density
        n_plus(rho) = sum_s Z_s n_s(rho) = Zbar_i(rho) * n_i(rho),
    where ``Zbar_i = sum_s Z_s f_s`` is the mean charge per ion (>= 1), so
    n_e = n_plus = Zbar_i * n_i, i.e. n_i = n_e / Zbar_i.  Only a singly-charged
    plasma (Zbar_i = 1) gives n_i = n_e; with any multiply-charged ion the ions
    are diluted relative to the electrons, and that dilution is Zbar_i itself.

    Reads the ``Zbar_i`` PROFILE (mean ion charge from composition, uniform-at-
    average until the species densities carry their own radial shapes).  It is a
    weak *default*, so an independent ion profile (its own Angioni peaking +
    generator) pre-empts it -- quasineutrality does not have to win over shaping.
    Where it IS the active producer the ion shape follows the electron shape,
    correctly diluted by Zbar_i, and the charge imbalance ``chi_e = n_e/n_plus``
    reads 1; where Angioni shapes n_i independently, chi_e departs from 1.

    MEASURED (2026-08-04): it must divide by Zbar_i, not collapse to a bare
    ``n_i = n_e``.  ``n_i = n_e`` IS the statement Zbar_i = 1 = no dilution, and
    propagated to the level by the ion volume-average consistency it fights the
    dilution anchor -- PROCESS_large_tokamak fails 22 tests, P_fus +20.8%.  The
    dilution is a LEVEL effect (n_i_avg < n_e_avg) present even when the ion and
    electron SHAPES match, so it cannot be dropped from the pointwise tie.  See
    .claude/scratchpad.md.
    """
    return n_e / _positive_denominator(Zbar_i, name="mean ion charge")


@relation(
    name="Mean ion charge from composition",
    tags=("plasma", "composition"),
    outputs="Zbar_i_avg",
)
def mean_ion_charge_from_composition(
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_p: Any = 0.0,
    c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0, c_O: Any = 0.0,
    c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
    Zbar_Li: Any = 3.0, Zbar_Be: Any = 4.0, Zbar_C: Any = 6.0, Zbar_N: Any = 7.0,
    Zbar_O: Any = 8.0, Zbar_Ne: Any = 10.0, Zbar_Ar: Any = 18.0, Zbar_Kr: Any = 36.0,
    Zbar_Xe: Any = 54.0, Zbar_W: Any = 74.0,
) -> Any:
    """chi_e = n_e/n_i = sum_s Z_s f_s, the mean charge per ion.

    Derived rather than postulated.  Splitting the species sum into the ones
    carried as fractions of n_i (the fuel and ash) and the ones carried as
    concentrations relative to n_e (the impurities, f_z = c_z * chi_e):

        chi_e = [f_D + f_T + f_p + 2(f_He3 + f_He4)]  +  chi_e * sum_z c_z Zbar_z

    and solving for chi_e gives

        chi_e = zbar_fuel / (1 - sum_z c_z Zbar_z)

    i.e. exactly the fuel-only mean charge the composition relations already
    compute, DIVIDED BY the impurity term every one of them drops.  With no
    impurities the denominator is 1 and chi_e degenerates to zbar_fuel, so a
    hydrogenic plasma still gives chi_e = 1 and n_i = n_e.

    Hydrogen enters at Z = 1 and helium at Z = 2 (both fully stripped at any
    temperature fusdb models); the impurities bring their own ``Zbar_X``, so
    chi_e carries a temperature dependence through them and is not a pure
    composition quantity.

    Helium is NOT in the impurity sum -- it is already in the f-fractions above
    and would otherwise be counted twice.
    """
    zbar_fuel = f_D + f_T + f_p + 2.0 * (f_He3 + f_He4)
    concentrations = {"Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    charges = {"Li": Zbar_Li, "Be": Zbar_Be, "C": Zbar_C, "N": Zbar_N, "O": Zbar_O,
               "Ne": Zbar_Ne, "Ar": Zbar_Ar, "Kr": Zbar_Kr, "Xe": Zbar_Xe, "W": Zbar_W}
    impurity_charge = 0.0
    for symbol, concentration in concentrations.items():
        impurity_charge = impurity_charge + concentration * charges[symbol]
    return zbar_fuel / _positive_denominator(1.0 - impurity_charge, name="hydrogenic electron fraction")


@relation(
    name="Fuel dilution from fuel fractions",
    tags=("plasma", "composition"),
    outputs="dilution",
)
def fuel_dilution_from_fuel_fractions(
    f_D: Any,
    f_T: Any,
    Zbar_i_avg: Any,
    f_He3: Any = 0.0,
) -> Any:
    """Fuel dilution = (n_D + n_T + 2 n_He3) / n_e -- the fuel share of the electrons.

    Charge-weighted, so He3 enters at Z = 2: the quantity answers "what
    fraction of the electrons is balanced by FUEL ions", which is what dilutes
    the fusion rate.  Written with fractions over ``chi_e`` because
    ``c_s = f_s / chi_e``:

        (f_D + f_T + 2 f_He3) * n_i/n_e  =  (f_D + f_T + 2 f_He3) / Zbar_i_avg

    It counts He3 as FUEL and protium as ASH, so it is NOT the hydrogenic
    concentration ``c_H`` = (p + D + T)/n_e.  The two coincide only when there
    is no protium and no He3 -- every D-T case in the suite -- so the
    difference cannot be caught by the current tests and has to be kept
    straight by definition.

    ``Fuel dilution from impurity concentrations (cfspopcon)`` is the other
    producer: cfspopcon counts only D and T and reaches them from the impurity
    side, ``1 - sum_X c_X Zbar_X``.  That is exact under cfspopcon's own
    assumptions (no He3 fuel, all helium an impurity) but a simplification
    here, so this relation is the default and cfspopcon's is selected by name
    in tests/cfspopcon_SPARC.
    """
    return (f_D + f_T + 2.0 * f_He3) / _positive_denominator(Zbar_i_avg, name="mean ion charge")


@relation(
    name="Positive charge density from species",
    tags=("plasma", "composition"),
    outputs="n_plus",
)
def positive_charge_density_from_species(
    n_D: Any,
    n_T: Any,
    n_He3: Any,
    n_He4: Any,
    n_e: Any,
    n_p: Any = 0.0,
    c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0, c_O: Any = 0.0,
    c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
    *,
    Zbar_Li: Any, Zbar_Be: Any, Zbar_C: Any, Zbar_N: Any,
    Zbar_O: Any, Zbar_Ne: Any, Zbar_Ar: Any, Zbar_Kr: Any,
    Zbar_Xe: Any, Zbar_W: Any,
) -> Any:
    """n_plus(rho) = sum_s Z_s n_s(rho) -- the positive-charge density.

    Fuel ions at their nuclear charge (D, T, p at Z=1; He3, He4 at Z=2) plus the
    impurities (c_k * Zbar_k * n_e).  Helium is counted ONCE, through n_He3/n_He4
    (the fusion ash), so it is excluded from the impurity sum -- the same
    convention as ``Mean ion charge from composition``.  Quasineutrality is the
    CHARGE balance n_e = n_plus; the ratio n_e/n_plus is the charge imbalance
    ``chi_e``.

    Computed from the ACTUAL species densities, so when an independent ion shape
    (Angioni) pulls n_i off the electron profile the positive charge tracks it
    and the imbalance departs from 1.  ``Zbar_k`` are real inputs (Mavrin coronal
    charges), not bare-atomic-number constants -- see impurities.py.
    """
    fuel_charge = n_D + n_T + n_p + 2.0 * (n_He3 + n_He4)
    concentrations = {"Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    charges = {"Li": Zbar_Li, "Be": Zbar_Be, "C": Zbar_C, "N": Zbar_N, "O": Zbar_O,
               "Ne": Zbar_Ne, "Ar": Zbar_Ar, "Kr": Zbar_Kr, "Xe": Zbar_Xe, "W": Zbar_W}
    impurity_charge_fraction = 0.0
    for symbol, concentration in concentrations.items():
        impurity_charge_fraction = impurity_charge_fraction + concentration * charges[symbol]
    return fuel_charge + n_e * impurity_charge_fraction


@relation(
    name="Charge imbalance from densities",
    tags=("plasma", "composition"),
    outputs="chi_e",
)
def charge_imbalance_from_densities(
    n_e: Any,
    n_plus: Any,
) -> Any:
    """chi_e(rho) = n_e(rho) / n_plus(rho) -- the charge-imbalance ratio.

    Quasineutrality is n_e = n_plus, so this reads 1 wherever it holds -- which
    is everywhere the ions are built to match the electrons (the quasineutrality
    default n_i = n_e/Zbar_i).  A pure DIAGNOSTIC: nothing consumes it, so it
    cannot draw composition into a density-ratio cycle.  It departs from 1 only
    where an independent ion shape (Angioni peaking) makes the positive-charge
    density track the electrons imperfectly along the profile -- the residual
    local charge imbalance.
    """
    return n_e / _positive_denominator(n_plus, name="positive charge density")
