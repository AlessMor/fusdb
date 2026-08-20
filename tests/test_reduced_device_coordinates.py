import numpy as np

from fusdb.profiles.system import build_relation_system
from fusdb.registry import RELATIONS, TAGS, VARIABLES
from fusdb.variable import Variable


def _selected_names(device: str) -> set[str]:
    return {
        relation.name
        for relation in RELATIONS.get_filtered_relations(tags=TAGS.expand((device,)))
    }


def _mapping_system(*names: str):
    relations = [RELATIONS.get(name) for name in names]
    system = build_relation_system([], relations, profile_size=46).compile()
    system.pack()
    values = system.complete(system.solver_values())
    return system, values


def test_stellarator_defaults_are_device_local():
    names = _selected_names("stellarator")
    assert "Reduced stellarator toroidal-flux coordinate" in names
    assert "Reduced stellarator normalized enclosed volume" in names
    assert "Reduced stellarator volume integration weight" in names
    assert "Tokamak normalized enclosed volume" not in names
    assert "Reduced mirror normalized enclosed volume" not in names


def test_mirror_defaults_are_device_local():
    names = _selected_names("mirror")
    assert "Reduced mirror radial coordinate" in names
    assert "Reduced mirror normalized enclosed volume" in names
    assert "Reduced mirror volume integration weight" in names
    assert "Tokamak normalized enclosed volume" not in names
    assert "Reduced stellarator normalized enclosed volume" not in names


def test_reduced_stellarator_contract_is_deterministic_and_nonpacked():
    system, values = _mapping_system(
        "Reduced stellarator toroidal-flux coordinate",
        "Reduced stellarator normalized enclosed volume",
        "Reduced stellarator volume integration weight",
    )
    rho = np.asarray(values["rho"], dtype=float)
    np.testing.assert_allclose(values["rho_tor"], rho)
    np.testing.assert_allclose(values["v_norm"], rho**2)
    np.testing.assert_allclose(values["w_V"], rho)
    for name in ("rho_tor", "v_norm", "w_V"):
        assert name not in system.packed_variables
        assert all(spec[0] != name for spec in system.packed_specs)


def test_reduced_mirror_contract_keeps_axial_physics_out_of_rho():
    system, values = _mapping_system(
        "Reduced mirror radial coordinate",
        "Reduced mirror normalized enclosed volume",
        "Reduced mirror volume integration weight",
    )
    rho = np.asarray(values["rho"], dtype=float)
    np.testing.assert_allclose(values["rho_radial"], rho)
    np.testing.assert_allclose(values["v_norm"], rho**2)
    np.testing.assert_allclose(values["w_V"], rho)
    for name in ("rho_radial", "v_norm", "w_V"):
        assert name not in system.packed_variables
        assert all(spec[0] != name for spec in system.packed_specs)


def test_supplied_equilibrium_mapping_suppresses_reduced_default_provider():
    supplied = np.linspace(0.0, 1.0, 46) ** 1.25
    fallback = RELATIONS.get("Reduced stellarator toroidal-flux coordinate")
    system = build_relation_system(
        [Variable("rho_tor", value=supplied)],
        [fallback],
        profile_size=46,
    ).compile()
    assert fallback.name not in {relation.name for relation in system.model.candidate_primary_relations}
    system.pack()
    values = system.complete(system.solver_values())
    np.testing.assert_allclose(values["rho_tor"], supplied)
    assert "rho_tor" in system.fixed
    assert "rho_tor" not in system.packed_variables


def test_coordinate_registry_exposes_reduced_device_mappings():
    assert "rho_radial" in VARIABLES
    assert set(VARIABLES.get("rho_tor").default_relation) == {
        "Tokamak normalized toroidal-flux coordinate",
        "Reduced stellarator toroidal-flux coordinate",
    }
    assert VARIABLES.get("rho_radial").default_relation == (
        "Reduced mirror radial coordinate",
    )
    assert set(VARIABLES.get("w_V").default_relation) == {
        "Tokamak volume integration weight",
        "Reduced stellarator volume integration weight",
        "Reduced mirror volume integration weight",
    }
