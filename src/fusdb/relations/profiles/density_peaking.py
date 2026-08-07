"""Angioni density-peaking scaling relations.

fusdb keeps separate electron and ion density peaking factors (``density_peaking``
and ``ion_density_peaking``). The electron Angioni variant produces
``density_peaking`` (the default producer). The ion Angioni variant produces
``ion_density_peaking``; it is off by default (``ion_density_peaking`` defaults to
the electron value) and active only when explicitly included. The ion variant
additionally needs ``dilution`` (the fuel share n_fuel/n_e), which fusdb's composition layer or the
user supplies.
"""

from typing import Any

import numpy as np

from fusdb.relation import relation

_N19 = 1.0e19  # cfspopcon expresses average_electron_density in 1e19 m^-3


def calc_density_peaking(effective_collisionality: Any, beta_T: Any, nu_noffset: Any) -> Any:
    """Calculate the density peaking (peak over volume average).

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Equation 3 from p1334 of Angioni et al :cite:`angioni_scaling_2007`.

    cfspopcon clips this to ``>= 1``; fusdb carries that bound on the variable
    instead (``density_peaking``/``ion_density_peaking`` domain ``[1, inf)``).
    A clip inside the formula makes the relation **non-invertible**: every
    ``nu_noffset`` below the clip point yields the same output, so solving the
    relation for the offset has infinitely many roots and the scalar scan
    returns the first grid point it tries -- which is ``-1e240``.  That is what
    broke DEMO_2022 (see NOTES.md).
    """
    return (1.347 - 0.117 * np.log(effective_collisionality) - 4.03 * beta_T) + nu_noffset


@relation(
    name="Effective collisionality (Angioni)",
    tags=("plasma", "profile", "tokamak"),
    outputs="effective_collisionality",
)
def calc_effective_collisionality(n_e_avg: Any, T_e_avg: Any, R: Any, Z_eff: Any) -> Any:
    """Calculate the effective collisionality.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    From p1327 of Angioni et al :cite:`angioni_scaling_2007`.

    Args:
        n_e_avg: [1/m^3] :term:`glossary link<average_electron_density>`
        T_e_avg: [keV] :term:`glossary link<average_electron_temp>`
        R: [m] :term:`glossary link<major_radius>`
        Z_eff: [~] :term:`glossary link<z_effective>`

    Returns:
        effective_collisionality [~]
    """
    # CHECK
    average_electron_density = n_e_avg / _N19
    return (0.1 * Z_eff * average_electron_density * R) / (T_e_avg**2.0)


@relation(
    name="Electron density peaking (Angioni)",
    tags=("default", "plasma", "profile", "tokamak"),
    outputs="density_peaking",
    # Warned, not enforced: a HOLLOW profile (peaking below 1) is physical --
    # it occurs at the very centre of the core -- so it must not fail the
    # reconcile.  But it is unusual enough to surface, and a reactor that
    # wants it rejected can re-declare the same constraint enforced.
    constraints=(("density_peaking >= 1.0", False),),
)
def calc_electron_density_peaking(
    effective_collisionality: Any, beta_T: Any, electron_density_peaking_offset: Any
) -> Any:
    """Calculate the electron density peaking.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Args:
        effective_collisionality: [~] :term:`glossary link<effective_collisionality>`
        beta_T: [~] :term:`glossary link<beta_toroidal>`
        electron_density_peaking_offset: [~] :term:`glossary link<electron_density_peaking_offset>`
    Returns:
        density_peaking
    """
    # CHECK
    return calc_density_peaking(effective_collisionality, beta_T, nu_noffset=electron_density_peaking_offset)


@relation(
    name="Ion density peaking (Angioni)",
    tags=("default", "plasma", "profile", "tokamak"),
    outputs="ion_density_peaking",
    # Warned, not enforced: a HOLLOW profile (peaking below 1) is physical --
    # it occurs at the very centre of the core -- so it must not fail the
    # reconcile.  But it is unusual enough to surface, and a reactor that
    # wants it rejected can re-declare the same constraint enforced.
    constraints=(("ion_density_peaking >= 1.0", False),),
)
def calc_ion_density_peaking(
    effective_collisionality: Any, beta_T: Any, ion_density_peaking_offset: Any
) -> Any:
    """Calculate the ion density peaking.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Producer of ``ion_density_peaking``; off by default -- not listed in
    ``ion_density_peaking.default_relation`` (which defaults the ion peaking to the
    electron value) -- and used only when explicitly included.

    Args:
        effective_collisionality: [~] :term:`glossary link<effective_collisionality>`
        beta_T: [~] :term:`glossary link<beta_toroidal>`
        ion_density_peaking_offset: [~] :term:`glossary link<ion_density_peaking_offset>`
    Returns:
        ion_density_peaking
    """
    # CHECK
    return calc_density_peaking(effective_collisionality, beta_T, nu_noffset=ion_density_peaking_offset)
