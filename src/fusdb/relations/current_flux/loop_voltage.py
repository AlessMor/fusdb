"""Loop-voltage relation."""

from typing import Any

from fusdb import relation


@relation(
    name="Loop voltage at flat-top",
    tags=("plasma", "current_drive", "tokamak"),
    outputs="loop_voltage",
)
def calc_loop_voltage(
    R: Any, a: Any, inductive_plasma_current: Any, kappa: Any, neoclassical_loop_resistivity: Any
) -> Any:
    """Calculate plasma toroidal loop voltage at flattop.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Plasma loop voltage from Alex Creely's original work.

    Args:
        R: [m] :term:`glossary link<major_radius>`
        a: [m] :term:`glossary link<minor_radius>`
        inductive_plasma_current: [A] :term:`glossary link<inductive_plasma_current>`
        kappa: [~] :term:`glossary link<areal_elongation>`
        neoclassical_loop_resistivity: [Ohm-m] :term:`glossary link<neoclassical_loop_resistivity>`

    Returns:
        loop_voltage [V]
    """
    # CHECK
    # Toroidal length over plasma cross-section surface area [1/m]
    _term1 = 2 * R / (a**2 * kappa)
    return inductive_plasma_current * _term1 * neoclassical_loop_resistivity
