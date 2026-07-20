"""Angioni density-peaking scaling relations.

fusdb keeps separate electron and ion density peaking factors (``density_peaking``
and ``ion_density_peaking``). The electron Angioni variant produces
``density_peaking`` (the default producer). The ion Angioni variant produces
``ion_density_peaking``; it is off by default (``ion_density_peaking`` defaults to
the electron value) and active only when explicitly included. The ion variant
additionally needs ``dilution`` (n_i/n_e), which fusdb's composition layer or the
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
    """
    nu_n = (1.347 - 0.117 * np.log(effective_collisionality) - 4.03 * beta_T) + nu_noffset
    return np.maximum(nu_n, 1.0)


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
