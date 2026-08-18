import numpy as np

from fusdb.profiles.system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.variable import Variable


PRF = "PRF electron temperature profile"
MINOR = "Tokamak normalized minor-radius coordinate"
WEIGHT = "Tokamak volume integration weight"


def _prf_relation(system):
    return next(rel for rel in system.model.candidate_primary_relations if rel.name == PRF)


def _prf_profile(rho_minor):
    system = build_relation_system(
        [
            Variable("T_e_avg", value=8.0),
            Variable("temperature_peaking", value=2.0),
            Variable("rho_minor", value=rho_minor),
            Variable("w_V", value=np.linspace(0.0, 1.0, 31) ** 1.25),
        ],
        (RELATIONS.get(PRF),),
        profile_size=31,
    ).compile()
    return system, np.asarray(system.complete(system.solver_values())["T_e"]).reshape(-1)


def test_compilation_does_not_rewrite_a_relation_declaration():
    """A Relation describes its equation, not one scenario's compiled form.

    The same registry object must come out of two structurally different
    systems -- supplied mapping and materialized tokamak fallback -- with one
    dependency declaration, so inspecting a relation in the registry and inside
    a reactor shows the same thing.
    """
    declared = RELATIONS.get(PRF)
    supplied, _profile = _prf_profile(np.linspace(0.0, 1.0, 31) ** 1.1)
    fallback = build_relation_system(
        [Variable("T_e_avg", value=8.0), Variable("temperature_peaking", value=2.0)],
        (RELATIONS.get(MINOR), RELATIONS.get(WEIGHT), RELATIONS.get(PRF)),
        profile_size=31,
    ).compile()

    for system in (supplied, fallback):
        relation = _prf_relation(system)
        assert relation.input_names == declared.input_names
        assert relation.constant_names == declared.constant_names
        assert relation.argument_names == declared.argument_names


def test_prf_uses_supplied_minor_radius_and_volume_measure_as_dependencies():
    """A supplied mapping must reach the physics, not be shadowed by a default."""
    system, straight = _prf_profile(np.linspace(0.0, 1.0, 31))
    _system, warped = _prf_profile(np.linspace(0.0, 1.0, 31) ** 1.1)

    assert "rho_minor" in system.unresolved_dependencies(_prf_relation(system))
    assert not np.allclose(straight, warped)


def test_prf_static_tokamak_coordinate_defaults_keep_fast_constant_path():
    """A materialized fallback mapping must add no solver ancestry.

    It stays a declared argument of the relation, but compilation knows it is
    already resolved: it is not an active variable, so it contributes no domain
    rows, and it does not disqualify the relation as a profile generator.
    """
    system = build_relation_system(
        [Variable("T_e_avg", value=8.0), Variable("temperature_peaking", value=2.0)],
        (RELATIONS.get(MINOR), RELATIONS.get(WEIGHT), RELATIONS.get(PRF)),
        profile_size=31,
    ).compile()

    relation = _prf_relation(system)
    assert "rho_minor" in relation.input_names
    assert "rho_minor" not in system.unresolved_dependencies(relation)
    assert system.variable_roles["rho_minor"] == "inactive"

    values = system.complete(system.solver_values())
    layout = system.residual_layout(values)
    assert "rho_minor" not in {name for name, _j, _rows in layout["domain_tail"]}
    assert "w_V" not in {name for name, _j, _rows in layout["domain_tail"]}
    assert system.profile_average_by_name.get("T_e") == "T_e_avg"
