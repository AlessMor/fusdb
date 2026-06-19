"""Loop-voltage and resistivity relations."""

def calc_loop_voltage(major_radius, minor_radius, inductive_plasma_current, areal_elongation, neoclassical_loop_resistivity):
    """cfspopcon: plasma toroidal loop voltage at flattop."""
    Iind = inductive_plasma_current
    _term1 = 2 * major_radius / (minor_radius**2 * areal_elongation)
    return Iind * _term1 * neoclassical_loop_resistivity
