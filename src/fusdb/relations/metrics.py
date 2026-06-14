# TODO(low): add from cfspopcon
    # Larmor radius for the species (rho)
    # Normalized gyroradius (rho*)
    # Collisionality (edge, normalised and effective)
    # e-e and e-i collision frequencies
    # turbulence parameter
#
# cfspopcon ports (UNDECORATED scaffolds), source:
# cfspopcon/formulas/metrics/{collisionality.py,larmor_radius.py}
# Review formula + variable name/unit mapping (e.g. average_electron_density->n_avg,
# average_ion_temp->T_i_avg, magnetic_field_on_axis->B0, minor_radius->a, q_star,
# z_effective->Z_eff), then add @relation to activate. The collisionality formulas
# below reference cfspopcon's pint unit registry (ureg/convert_units) and helper
# collision-frequency functions; wire those to fusdb constants before activating.

import numpy as np

# Vacuum/electron constants (SI) for the ported Larmor-radius formula.
_ELEMENTARY_CHARGE = 1.602176634e-19  # C


# TODO(cfspopcon): helper for the collisionality formulas (Coulomb logarithm).
def calc_coulomb_logarithm(electron_density, electron_temp):
    """cfspopcon: Coulomb logarithm (Verdoolaege 2021). density [m^-3], temp [eV]."""
    return 30.9 - np.log(electron_density**0.5 * electron_temp**-1.0)


# TODO(cfspopcon): activate as a fusdb relation (nu_star output).
#   References cfspopcon ureg.e / ureg.epsilon_0 / convert_units — wire to fusdb units.
def calc_normalised_collisionality(
    average_electron_density,
    average_electron_temp,
    average_ion_temp,
    q_star,
    major_radius,
    inverse_aspect_ratio,
    z_effective,
):
    """cfspopcon: normalized collisionality nu_star (Verdoolaege 2021 Eq. 1c)."""
    from cfspopcon.unit_handling import convert_units, ureg  # noqa: F401  # TODO: replace with fusdb units

    return convert_units(
        ureg.e**4
        / (2.0 * np.pi * 3**1.5 * ureg.epsilon_0**2)
        * calc_coulomb_logarithm(average_electron_density, average_electron_temp)
        * average_electron_density
        * q_star
        * major_radius
        * z_effective
        / (average_ion_temp**2 * inverse_aspect_ratio**1.5),
        ureg.dimensionless,
    )


# TODO(cfspopcon): helper for rho_star (Larmor radius, Eich 2020 Eq.1).
def calc_larmor_radius(species_temperature, magnetic_field_strength, species_mass):
    """cfspopcon: Larmor radius = sqrt(T*m)/(e*B)."""
    return np.sqrt(species_temperature * species_mass) / (_ELEMENTARY_CHARGE * magnetic_field_strength)


# TODO(cfspopcon): activate as a fusdb relation (rho_star output).
def calc_rho_star(average_ion_mass, average_ion_temp, magnetic_field_on_axis, minor_radius):
    """cfspopcon: rho* (normalized gyroradius, Verdoolaege 2021 Eq. 1a)."""
    rho_s = calc_larmor_radius(
        species_temperature=average_ion_temp, magnetic_field_strength=magnetic_field_on_axis, species_mass=average_ion_mass
    )
    return rho_s / minor_radius
