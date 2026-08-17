import numpy as np

from fusdb.profile_system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.variable import Variable


PRF = "PRF electron temperature profile"
MINOR = "Tokamak normalized minor-radius coordinate"
WEIGHT = "Tokamak volume integration weight"


def _prf_relation(system):
    return next(rel for rel in system.model.candidate_primary_relations if rel.name == PRF)


def test_prf_uses_supplied_minor_radius_and_volume_measure_as_dependencies():
    rho_minor = np.linspace(0.0, 1.0, 31) ** 1.1
    w_V = np.linspace(0.0, 1.0, 31) ** 1.25
    system = build_relation_system(
        [
            Variable("T_e_avg", value=8.0),
            Variable("temperature_peaking", value=2.0),
            Variable("rho_minor", value=rho_minor),
            Variable("w_V", value=w_V),
        ],
        (RELATIONS.get(PRF),),
        profile_size=31,
    ).compile()

    relation = _prf_relation(system)
    assert "rho_minor" in relation.input_names
    assert "w_V" in relation.input_names
    assert "rho" in relation.constant_names


def test_prf_static_tokamak_coordinate_defaults_keep_fast_constant_path():
    system = build_relation_system(
        [
            Variable("T_e_avg", value=8.0),
            Variable("temperature_peaking", value=2.0),
        ],
        (RELATIONS.get(MINOR), RELATIONS.get(WEIGHT), RELATIONS.get(PRF)),
        profile_size=31,
    ).compile()

    relation = _prf_relation(system)
    assert "rho_minor" not in relation.input_names
    assert "w_V" not in relation.input_names
    assert "rho_minor" in relation.constant_names
    assert "w_V" in relation.constant_names
