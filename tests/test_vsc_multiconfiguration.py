import numpy as np

from fusdb.registry import RELATIONS, TAGS, VARIABLES, MU0
from fusdb.registry.coordinate_variables import PHYSICAL_COORDINATE_NAMES
from fusdb.relations.confinement.balance import thermal_stored_energy_vsc_profile
from fusdb.relations.confinement.scalings.frc import (
    frc_lsx_confinement_time,
    frc_resistive_diffusion_time,
)
from fusdb.relations.geometry.dipole_geometry import (
    finite_dipole_ring_current,
    point_dipole_normalized_u_coordinate,
    point_dipole_plasma_volume,
)
from fusdb.relations.geometry.frc_geometry import frc_superellipse_plasma_volume
from fusdb.relations.geometry.mirror_geometry import (
    mirror_diamagnetic_central_field,
    mirror_sin2_volume,
)
from fusdb.relations.plasma_state.frc_equilibrium import frc_rigid_rotor_parameter


def test_vsc_device_tags_and_variables_are_registered():
    assert {"frc", "dipole"} <= TAGS.allowed
    for name in ("rho_vol", "rho_U", "G_B25", "a_c", "K_frc", "L_shell", "iota_2_3"):
        assert name in VARIABLES
    assert {"rho_vol", "rho_U"} <= PHYSICAL_COORDINATE_NAMES


def test_original_stored_energy_is_default_and_vsc_form_is_alternative():
    assert VARIABLES.get("W_th").default_relation == ("Thermal stored energy",)
    active = {rel.name for rel in RELATIONS.get_filtered_relations(tags=("frc",))}
    assert "Thermal stored energy" in active
    assert "Thermal stored energy (VSC profile model)" not in active
    value = thermal_stored_energy_vsc_profile.func(2e20, 10.0, 2e20, 10.0, 1.0, 1.0, 10.0)
    assert value > 0.0


def test_alternative_w_th_provider_is_gated_by_default_relation_not_by_a_tag():
    """The VSC W_th form is held back by the producer whitelist, not an opt-in tag.

    It is an ordinary secondary provider: a variable's ``default_relation``
    already filters every producer that is not on it.
    """
    blocked = "Thermal stored energy (VSC profile model)"
    active = {rel.name for rel in RELATIONS.get_filtered_relations(tags=TAGS.expand(("tokamak",)))}
    assert blocked in {rel.name for rel in RELATIONS.producers("W_th")}
    assert blocked not in VARIABLES.get("W_th").default_relation
    assert blocked not in active


def test_mirror_geometry_equations_match_published_forms():
    beta = 0.2
    bc = mirror_diamagnetic_central_field.func(4.0, beta)
    assert np.isclose(bc, 4.0 * np.sqrt(0.8))
    rmc = 10.0 / np.sqrt(0.8)
    volume = mirror_sin2_volume.func(0.5, 5.0, 1.0, rmc)
    expected = np.pi * 0.5**2 * (5.0 + 2.0 / np.sqrt(rmc))
    assert np.isclose(volume, expected)


def test_frc_rigid_rotor_and_geometry_equations():
    xs = 0.6
    k = float(frc_rigid_rotor_parameter.func(xs))
    assert np.isclose(np.tanh(k) / k, 1.0 - xs**2 / 2.0, rtol=1e-10)
    volume = frc_superellipse_plasma_volume.func(0.5, 3.0, 2.0)
    assert np.isclose(volume, (2.0 / 3.0) * np.pi * 0.5**2 * 3.0)
    tau_eta = frc_resistive_diffusion_time.func(0.5, 1e-7)
    assert np.isclose(tau_eta, MU0 * 0.5**2 / 1e-7)
    tau_lsx = frc_lsx_confinement_time.func(3.0, xs, 0.5, 1e21)
    assert np.isclose(tau_lsx, 3.2e-15 * 3.0**0.5 * xs**0.8 * 0.5**2.1 * (1e21)**0.6)


def test_point_dipole_equations():
    volume = point_dipole_plasma_volume.func(1.0, 2.0)
    assert np.isclose(volume, (64.0 * np.pi / 105.0) * (8.0 - 1.0))
    current = finite_dipole_ring_current.func(5.0, 0.5)
    assert np.isclose(MU0 * current, 4.0 * 0.5 * 5.0)
    rho = np.linspace(0.0, 1.0, 5)
    rho_u = point_dipole_normalized_u_coordinate.func(1.0, 2.0, rho=rho)
    assert np.isclose(rho_u[0], 0.0)
    assert np.isclose(rho_u[-1], 1.0)


def test_vsc_relations_discover_cleanly():
    names = {rel.name for rel in RELATIONS}
    assert "Mirror self-consistent transport loss" in names
    assert "FRC LSX confinement time" in names
    assert "Point-dipole plasma volume" in names
    assert "Near-axis stellarator B2.5 moment" in names
    assert "p-B11 reaction rate" in names
