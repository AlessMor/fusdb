import numpy as np

from fusdb.profile_system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.variable import Variable


def test_static_tokamak_coordinate_defaults_are_materialized_not_providers():
    relations = tuple(
        RELATIONS.get(name)
        for name in (
            "Tokamak normalized minor-radius coordinate",
            "Tokamak normalized enclosed volume",
            "Tokamak volume integration weight",
        )
    )
    system = build_relation_system([], relations, profile_size=46)

    candidate_names = {relation.name for relation in system.candidate_primary_relations}
    assert "Tokamak normalized minor-radius coordinate" not in candidate_names
    assert "Tokamak normalized enclosed volume" not in candidate_names
    assert "Tokamak volume integration weight" not in candidate_names

    rho = np.linspace(0.0, 1.0, 46)
    assert np.array_equal(system.inputs["rho_minor"], rho)
    assert np.array_equal(system.inputs["w_V"], rho)
    assert np.array_equal(system.inputs["v_norm"], rho**2)
    assert {"rho_minor", "v_norm", "w_V"} <= system.fixed


def test_geometry_dependent_coordinate_provider_remains_a_relation():
    relation = RELATIONS.get("Sauter self-similar profile volume mapping")
    system = build_relation_system(
        [Variable("delta", value=0.3), Variable("eps", value=0.3)],
        [relation],
        profile_size=46,
    )

    candidate_names = {candidate.name for candidate in system.candidate_primary_relations}
    assert relation.name in candidate_names
    assert system.inputs.get("v_norm") is None
    assert system.inputs.get("w_V") is None
