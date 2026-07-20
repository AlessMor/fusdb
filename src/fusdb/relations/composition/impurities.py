"""Impurity composition and dilution relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation


from ..utils import _species_fraction

@relation(
    name="Integrated Imp fraction from density profiles",
    tags=("plasma", "composition"),
    outputs="f_Imp",
)
def integrated_impurity_fraction_from_density_profiles(n_imp: Any, n_i: Any) -> Any:
    """Return pointwise impurity fraction from density profiles."""
    return _species_fraction(n_imp, n_i, name="f_Imp")


@relation(
    name="Impurity density from ion density and impurity fraction",
    tags=("plasma", "composition", "inverse"),
    outputs="n_imp",
)
def impurity_density_from_ion_density_and_fraction(n_i: Any, f_Imp: Any) -> Any:
    """Return impurity density from total ion density and impurity fraction."""
    return n_i * f_Imp

# ── Impurity-concentration composition (cfspopcon structure, Mavrin charge states) ──
# cfspopcon's calc_zeff_and_dilution_due_to_impurities builds Z_eff and the ion
# inventory from the supplied c_z = n_z/n_e concentrations and a T-dependent
# mean charge state per species (radas/ADAS interpolators there); these
# relations reproduce that structure with the Mavrin 2018 coronal mean-charge
# fits (the ``zc`` blocks of the mavrin_coronal datasets), so no external
# atomic data is needed.  Evaluated at the volume-averaged temperature, like
# cfspopcon (which feeds average_electron_temp to the interpolator).

_MAVRIN_CHARGE_SPECIES = ("He", "Li", "Be", "C", "N", "O", "Ne", "Ar", "Kr", "Xe", "W")


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
    T_e_avg: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """Z_eff = 1 + sum_z c_z * Zbar_z(T_e) * (Zbar_z(T_e) - 1), at T_e_avg.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    The starting Z_eff = 1 is the pure-hydrogenic plasma; each impurity adds
    c_z * Zbar * (Zbar - 1) (cfspopcon's ``calc_change_in_zeff``), with the
    mean charge from the Mavrin 2018 coronal fits instead of radas.
    """
    # CHECK
    concentrations = dict(zip(_MAVRIN_CHARGE_SPECIES,
                              (c_He, c_Li, c_Be, c_C, c_N, c_O, c_Ne, c_Ar, c_Kr, c_Xe, c_W)))
    z_effective = 1.0
    for concentration, zbar in _mavrin_charge_terms(T_e_avg, concentrations):
        z_effective = z_effective + concentration * zbar * (zbar - 1.0)
    return z_effective


@relation(
    name="Dilution from impurity concentrations (Mavrin)",
    tags=("plasma", "composition"),
    outputs="dilution",
)
def dilution_from_impurity_concentrations(
    T_e_avg: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """dilution = n_fuel / n_e = 1 - sum_z c_z * Zbar_z(T_e), at T_e_avg.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    cfspopcon's ``calc_zeff_and_dilution_due_to_impurities`` returns the
    fuel-ion dilution together with Z_eff; quasineutrality with hydrogenic fuel
    gives n_fuel = n_e - sum_z n_z * Zbar_z, i.e. the fuel fraction is
    1 - sum(c_z * Zbar_z).  This is the electron-balance complement of
    ``n_i_avg`` above (which adds the impurity ions themselves back in).  Mean
    charge from the Mavrin 2018 coronal fits instead of radas.
    """
    # CHECK
    concentrations = dict(zip(_MAVRIN_CHARGE_SPECIES,
                              (c_He, c_Li, c_Be, c_C, c_N, c_O, c_Ne, c_Ar, c_Kr, c_Xe, c_W)))
    dilution_terms = 0.0
    for concentration, zbar in _mavrin_charge_terms(T_e_avg, concentrations):
        dilution_terms = dilution_terms + concentration * zbar
    return np.maximum(1.0 - dilution_terms, 0.0)


@relation(
    name="Ion density average from impurity concentrations (Mavrin)",
    tags=("plasma", "composition"),
    outputs="n_i_avg",
)
def ion_density_average_from_impurity_concentrations(
    n_e_avg: Any,
    T_e_avg: Any,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0, c_W: Any = 0.0,
) -> Any:
    """n_i_avg = n_e_avg * (1 - sum_z c_z * (Zbar_z(T_e) - 1)), at T_e_avg.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    cfspopcon's dilution is n_fuel/n_e = 1 - sum(c_z * Zbar_z); fusdb's
    ``n_i_avg`` counts every ion species, so the impurities themselves are
    added back: n_i = n_fuel + n_e * sum(c_z) = n_e * (1 - sum(c_z*(Zbar-1))).
    Mean charge from the Mavrin 2018 coronal fits instead of radas.
    """
    # CHECK
    concentrations = dict(zip(_MAVRIN_CHARGE_SPECIES,
                              (c_He, c_Li, c_Be, c_C, c_N, c_O, c_Ne, c_Ar, c_Kr, c_Xe, c_W)))
    dilution_terms = 0.0
    for concentration, zbar in _mavrin_charge_terms(T_e_avg, concentrations):
        dilution_terms = dilution_terms + concentration * (zbar - 1.0)
    return n_e_avg * np.maximum(1.0 - dilution_terms, 0.0)
