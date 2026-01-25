"""Ion composition relations for densities and fractions."""

from __future__ import annotations

from fusdb.reactor_class import Reactor


@Reactor.relation(
    "plasma",
    name="Electron density from volume-averaged density",
    output="n_e",
    variables=("n_avg",),
)
def electron_density_from_average(n_avg: float) -> float:
    """Return electron density from volume-averaged density."""
    return n_avg


@Reactor.relation(
    "plasma",
    name="Electron density from ion fractions",
    output="n_e",
)
def electron_density_from_fractions(
    n_i: float,
    f_D: float,
    f_T: float,
    f_He3: float,
    f_He4: float,
) -> float:
    """Return electron density from ion density and ion fractions."""
    return n_i * (f_D + f_T + 2.0 * f_He3 + 2.0 * f_He4)
# TODO(low): improve the formulation so that it takes all ion fractions automatically


@Reactor.relation(
    "plasma",
    name="Deuterium density from fraction",
    output="n_D",
)
def deuterium_density(f_D: float, n_i: float) -> float:
    """Return deuterium density from ion fraction and total ion density."""
    return f_D * n_i


@Reactor.relation(
    "plasma",
    name="Tritium density from fraction",
    output="n_T",
)
def tritium_density(f_T: float, n_i: float) -> float:
    """Return tritium density from ion fraction and total ion density."""
    return f_T * n_i


@Reactor.relation(
    "plasma",
    name="Helium-3 density from fraction",
    output="n_He3",
)
def helium3_density(f_He3: float, n_i: float) -> float:
    """Return helium-3 density from ion fraction and total ion density."""
    return f_He3 * n_i


@Reactor.relation(
    "plasma",
    name="Helium-4 density from fraction",
    output="n_He4",
)
def helium4_density(f_He4: float, n_i: float) -> float:
    """Return helium-4 density from ion fraction and total ion density."""
    return f_He4 * n_i


@Reactor.relation(
    "plasma",
    name="Deuterium fraction from density",
    output="f_D",
    constraints=("n_i > 0",),
)
def deuterium_fraction(n_D: float, n_i: float) -> float:
    """Return deuterium fraction from density and total ion density."""
    return n_D / n_i


@Reactor.relation(
    "plasma",
    name="Tritium fraction from density",
    output="f_T",
    constraints=("n_i > 0",),
)
def tritium_fraction(n_T: float, n_i: float) -> float:
    """Return tritium fraction from density and total ion density."""
    return n_T / n_i


@Reactor.relation(
    "plasma",
    name="Helium-3 fraction from density",
    output="f_He3",
    constraints=("n_i > 0",),
)
def helium3_fraction(n_He3: float, n_i: float) -> float:
    """Return helium-3 fraction from density and total ion density."""
    return n_He3 / n_i


@Reactor.relation(
    "plasma",
    name="Helium-4 fraction from density",
    output="f_He4",
    constraints=("n_i > 0",),
)
def helium4_fraction(n_He4: float, n_i: float) -> float:
    """Return helium-4 fraction from density and total ion density."""
    return n_He4 / n_i


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_D)",
    output="f_D",
)
def deuterium_fraction_normalized(f_T: float, f_He3: float, f_He4: float) -> float:
    """Return deuterium fraction from the ion fraction normalization."""
    return 1.0 - f_T - f_He3 - f_He4


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_T)",
    output="f_T",
)
def tritium_fraction_normalized(f_D: float, f_He3: float, f_He4: float) -> float:
    """Return tritium fraction from the ion fraction normalization."""
    return 1.0 - f_D - f_He3 - f_He4


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_He3)",
    output="f_He3",
)
def helium3_fraction_normalized(f_D: float, f_T: float, f_He4: float) -> float:
    """Return helium-3 fraction from the ion fraction normalization."""
    return 1.0 - f_D - f_T - f_He4


@Reactor.relation(
    "plasma",
    name="Ion fraction normalization (solve f_He4)",
    output="f_He4",
)
def helium4_fraction_normalized(f_D: float, f_T: float, f_He3: float) -> float:
    """Return helium-4 fraction from the ion fraction normalization."""
    return 1.0 - f_D - f_T - f_He3


@Reactor.relation(
    "plasma",
    name="Average fuel mass number",
    output="afuel",
)
def average_fuel_mass_number(f_D: float, f_T: float, f_He3: float, f_He4: float) -> float:
    """Return average ion mass number from ion fractions."""
    return 2.0 * f_D + 3.0 * f_T + 3.0 * f_He3 + 4.0 * f_He4
# TODO(low): improve it to work automatically by taking data from allowed_species


# TODO(med): could add a relation to evaluate D, T, He3 injection terms at steady-state
    # Ndot_inj_x = V_p*(n_x/tau_p_x - n_x_production + n_x_burn)
def fuel_injection_terms():
    pass