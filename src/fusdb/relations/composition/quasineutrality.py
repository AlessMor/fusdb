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
    chi_e: Any,
) -> Any:
    """<n_i> = <n_e> / chi_e -- quasineutrality at the volume average.

    ``chi_e = n_e/n_i = sum_s Z_s f_s`` is the mean charge per ion, supplied by
    ``Mean ion charge from composition``.  This scalar anchor keeps the dilution
    physical even when the ion *profile shape* is derived independently of the
    electron one (its own peaking scaling + generator): the pointwise profile
    ties below are only weak defaults, so without this relation a decoupled ion
    side would float free of ``n_e``.

    It used to inline ``zbar = f_D + f_T + f_p + 2(f_He3 + f_He4)``, the
    FUEL-ONLY mean charge, which omits the impurity electrons entirely and so
    pinned n_i/n_e to 1 on any impurity-seeded machine without helium.  Reading
    ``chi_e`` instead is the whole of the fix: chi_e is that same zbar divided
    by ``1 - sum_z c_z Zbar_z``.  Measured on STELLARIS, the two differ by
    8.4e-04 -- exactly its charge-neutrality violation.
    """
    return n_e_avg / _positive_denominator(chi_e, name="mean ion charge")


@relation(
    name="Ion density from electron density and fuel fractions",
    tags=("default", "plasma", "composition", "inverse"),
    outputs="n_i",
)
def ion_density_from_electron_density_and_fuel_fractions(
    n_e: Any,
    chi_e: Any,
) -> Any:
    """n_i = n_e / chi_e, pointwise -- the profile twin of the anchor above.

    A weak *default*: applied pointwise it forces the ion profile shape to be
    proportional to the electron one (radially constant fractions), which is a
    reasonable fallback but must not pre-empt an independent derivation of the
    ion profile (own peaking scaling + generator, anchored by the
    averages-level quasineutrality above).

    ``chi_e`` replaces the inlined fuel-only ``zbar`` for the reason given
    above: zbar drops the impurity electrons, chi_e does not.
    """
    return n_e / _positive_denominator(chi_e, name="mean ion charge")


@relation(
    name="Mean ion charge from composition",
    tags=("plasma", "composition"),
    outputs="chi_e",
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
    chi_e: Any,
    f_He3: Any = 0.0,
) -> Any:
    """Fuel dilution = (n_D + n_T + 2 n_He3) / n_e -- the fuel share of the electrons.

    Charge-weighted, so He3 enters at Z = 2: the quantity answers "what
    fraction of the electrons is balanced by FUEL ions", which is what dilutes
    the fusion rate.  Written with fractions over ``chi_e`` because
    ``c_s = f_s / chi_e``:

        (f_D + f_T + 2 f_He3) * n_i/n_e  =  (f_D + f_T + 2 f_He3) / chi_e

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
    return (f_D + f_T + 2.0 * f_He3) / _positive_denominator(chi_e, name="mean ion charge")
