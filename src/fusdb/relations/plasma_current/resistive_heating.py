# TODO(high): from cfspopcon
    # P_ohm
    # Spitzer resistivity (with enhancement due to trapped particles)
    # current relaxation time?
    #loop voltage?
#
# cfspopcon ports (UNDECORATED scaffolds), source:
# cfspopcon/formulas/plasma_current/resistive_heating.py
# Review formula + variable name/unit mapping (e.g. inductive_plasma_current in MA,
# average_electron_temp->T_avg in keV, z_effective->Z_eff, P_ohmic->P_ohmic in W,
# major/minor_radius->R/a, areal_elongation->kappa) then add @relation to activate.


# TODO(cfspopcon): activate as a fusdb relation (P_ohmic output).
def calc_ohmic_power(inductive_plasma_current, loop_voltage):
    """cfspopcon: P_ohmic = inductive_plasma_current * loop_voltage."""
    return inductive_plasma_current * loop_voltage


# TODO(cfspopcon): activate as a fusdb relation (spitzer_resistivity output, Ohm-m).
def calc_Spitzer_loop_resistivity(average_electron_temp):
    """cfspopcon: parallel Spitzer loop resistivity (Coulomb log=17, Z=1; Wesson 2.16.2)."""
    return (2.8e-8) * (average_electron_temp ** (-1.5))


# TODO(cfspopcon): activate as a fusdb relation (trapped_particle_fraction output).
def calc_resistivity_trapped_enhancement(inverse_aspect_ratio, resistivity_trapped_enhancement_method=3):
    """cfspopcon: resistivity enhancement due to trapped particles (Wesson p801)."""
    if resistivity_trapped_enhancement_method == 1:
        trapped_particle_fraction = 1 / ((1.0 - (inverse_aspect_ratio**0.5)) ** 2.0)
    elif resistivity_trapped_enhancement_method == 2:
        trapped_particle_fraction = 2 / (1.0 - 1.31 * (inverse_aspect_ratio**0.5) + 0.46 * inverse_aspect_ratio)
    elif resistivity_trapped_enhancement_method == 3:
        trapped_particle_fraction = 0.609 / (0.609 - 0.785 * (inverse_aspect_ratio**0.5) + 0.269 * inverse_aspect_ratio)
    else:
        raise NotImplementedError(
            f"No implementation {resistivity_trapped_enhancement_method} for calc_resistivity_trapped_enhancement."
        )
    return trapped_particle_fraction


# TODO(cfspopcon): activate as a fusdb relation (neoclassical_loop_resistivity output, Ohm-m).
def calc_neoclassical_loop_resistivity(spitzer_resistivity, z_effective, trapped_particle_fraction):
    """cfspopcon: neoclassical loop resistivity incl. impurity ions (Wesson 14.10)."""
    return spitzer_resistivity * z_effective * 0.9 * trapped_particle_fraction


# TODO(cfspopcon): activate as a fusdb relation (loop_voltage output, V).
def calc_loop_voltage(major_radius, minor_radius, inductive_plasma_current, areal_elongation, neoclassical_loop_resistivity):
    """cfspopcon: plasma toroidal loop voltage at flattop."""
    Iind = inductive_plasma_current
    _term1 = 2 * major_radius / (minor_radius**2 * areal_elongation)
    return Iind * _term1 * neoclassical_loop_resistivity


# TODO(cfspopcon): activate as a fusdb relation (current_relaxation_time output, s).
def calc_current_relaxation_time(major_radius, inverse_aspect_ratio, areal_elongation, average_electron_temp, z_effective):
    """cfspopcon: current relaxation time (Bonoli). average_electron_temp in keV."""
    return 1.4 * ((major_radius * inverse_aspect_ratio) ** 2.0) * areal_elongation * (average_electron_temp**1.5) / z_effective
