"""Quasineutrality and composition helper relations."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry import SPECIES
from ..utils import _positive_denominator


_IMPURITY_CHARGE = float(SPECIES["Imp"].atomic_number)


@relation(
    name="Ion density from electron density and fuel fractions",
    tags=("plasma", "composition", "inverse"),
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
    """
    zbar = _positive_denominator(
        f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp,
        name="mean ion charge",
    )
    return n_e / zbar


@relation(
    name="Electron density from ion density and fuel fractions",
    tags=("plasma", "composition"),
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
    """
    zbar = _positive_denominator(
        f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4 + _IMPURITY_CHARGE * f_Imp,
        name="mean ion charge",
    )
    return n_i * zbar
