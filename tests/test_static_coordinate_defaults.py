import numpy as np

from fusdb.profiles.system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.variable import Variable


def test_static_tokamak_coordinate_defaults_are_materialized_not_providers():
    relations = tuple(
        RELATIONS.get(name)
        for name in (
            "Tokamak normalized minor-radius coordinate",
            "Tokamak normalized toroidal-flux coordinate",
            "Tokamak normalized enclosed volume",
            "Tokamak volume integration weight",
        )
    )
    system = build_relation_system([], relations, profile_size=46).compile()

    candidate_names = {relation.name for relation in system.model.candidate_primary_relations}
    assert "Tokamak normalized minor-radius coordinate" not in candidate_names
    assert "Tokamak normalized toroidal-flux coordinate" not in candidate_names
    assert "Tokamak normalized enclosed volume" not in candidate_names
    assert "Tokamak volume integration weight" not in candidate_names

    rho = np.linspace(0.0, 1.0, 46)
    assert np.array_equal(system.inputs["rho_minor"], rho)
    assert np.array_equal(system.inputs["rho_tor"], rho)
    assert np.array_equal(system.inputs["w_V"], rho)
    assert np.array_equal(system.inputs["v_norm"], rho**2)
    assert {"rho_minor", "rho_tor", "v_norm", "w_V"} <= system.fixed


def test_static_mapping_is_constant_but_dynamic_mapping_stays_input():
    avg = RELATIONS.get("Electron temperature volume-average consistency")
    static = RELATIONS.get("Tokamak volume integration weight")
    static_system = build_relation_system(
        [Variable("T_e_avg", value=10.0), Variable("T_e", value=np.full(46, 10.0), fixed=True)],
        [static, avg],
        profile_size=46,
    ).compile()
    static_relation = next(rel for rel in static_system.model.candidate_primary_relations if rel.name == avg.name)
    assert "w_V" in static_relation.input_names
    assert "w_V" not in static_system.unresolved_dependencies(static_relation)
    assert static_system.variable_roles["w_V"] == "inactive"

    dynamic = RELATIONS.get("Sauter self-similar profile volume mapping")
    dynamic_system = build_relation_system(
        [
            Variable("delta", value=0.3),
            Variable("eps", value=0.3),
            Variable("T_e_avg", value=10.0),
            Variable("T_e", value=np.full(46, 10.0), fixed=True),
        ],
        [dynamic, avg],
        profile_size=46,
    ).compile()
    dynamic_relation = next(rel for rel in dynamic_system.model.candidate_primary_relations if rel.name == avg.name)
    assert dynamic_relation.input_names == static_relation.input_names
    assert "w_V" in dynamic_system.unresolved_dependencies(dynamic_relation)
    assert dynamic_system.variable_roles["w_V"] == "computed"


def test_geometry_dependent_coordinate_provider_remains_a_relation():
    relation = RELATIONS.get("Sauter self-similar profile volume mapping")
    system = build_relation_system(
        [Variable("delta", value=0.3), Variable("eps", value=0.3)],
        [relation],
        profile_size=46,
    ).compile()

    candidate_names = {candidate.name for candidate in system.model.candidate_primary_relations}
    assert relation.name in candidate_names
    assert system.inputs.get("v_norm") is None
    assert system.inputs.get("w_V") is None
