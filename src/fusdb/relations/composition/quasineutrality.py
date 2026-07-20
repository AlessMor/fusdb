"""Quasineutrality and composition helper relations."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry import SPECIES
from ..utils import _positive_denominator


_IMPURITY_CHARGE = float(SPECIES["Imp"].atomic_number)


@relation(
    name="Ion density average from quasineutrality",
    tags=("plasma", "composition"),
    outputs="n_i_avg",
)
def ion_density_average_from_quasineutrality(
    n_e_avg: Any,
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return the volume-averaged ion density from quasineutrality.

    ``<n_i> = <n_e> / zbar`` with ``zbar = sum(f_X Z_X)`` -- quasineutrality
    integrated over the plasma volume with the (volume-averaged) ion
    fractions.  This scalar anchor keeps the dilution physical even when the
    ion *profile shape* is derived independently of the electron one (its own
    peaking scaling + generator): the pointwise profile ties below are only
    weak defaults, so without this relation a decoupled ion side would float
    free of ``n_e``.
    """
    zbar = _positive_denominator(
        f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp,
        name="mean ion charge",
    )
    return n_e_avg / zbar


@relation(
    name="Ion density from electron density and fuel fractions",
    tags=("default", "plasma", "composition", "inverse"),
    outputs="n_i",
)
def ion_density_from_electron_density_and_fuel_fractions(
    n_e: Any,
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return total ion density from electron density and ion fractions.

    This makes DT-only cases reachable from ``n_e``, ``f_D`` and ``f_T``.
    Additional He/impurity fractions are used when present in the namespace.

    A weak *default*: applied pointwise it forces the ion profile shape to be
    proportional to the electron one (radially constant fractions), which is a
    reasonable fallback but must not pre-empt an independent derivation of the
    ion profile (own peaking scaling + generator, anchored by the
    averages-level quasineutrality above).  The difference is carried by an
    implicit radial variation of the impurity fraction.
    """
    zbar = _positive_denominator(
        f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp,
        name="mean ion charge",
    )
    return n_e / zbar


@relation(
    name="Electron density from ion density and fuel fractions",
    tags=("default", "plasma", "composition"),
    outputs="n_e",
)
def electron_density_from_ion_density_and_fuel_fractions(
    n_i: Any,
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return electron density from ion density and ion fractions.

    This gives a direct consistency check for cases that supply both ``n_i``
    and ``n_e`` without requiring all individual minority densities.

    A weak *default*, for the same reason as the inverse relation above: the
    pointwise tie is a constant-fractions fallback, not a structural law.
    """
    zbar = _positive_denominator(
        f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp,
        name="mean ion charge",
    )
    return n_i * zbar


@relation(
    name="Effective charge from ion fractions",
    tags=("default", "plasma", "composition"),
    outputs="Z_eff",
)
def effective_charge_from_ion_fractions(
    f_D: Any,
    f_T: Any,
    f_He3: Any = 0.0,
    f_He4: Any = 0.0,
    f_Imp: Any = 0.0,
) -> Any:
    """Return Z_eff = sum(n_i Z_i^2) / n_e from ion fractions."""
    zbar = _positive_denominator(
        f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp,
        name="mean ion charge",
    )
    return (f_D + f_T + 4.0 * f_He3 + 4.0 * f_He4 + (_IMPURITY_CHARGE**2) * f_Imp) / zbar
