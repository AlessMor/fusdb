import numpy as np
import pytest

from fusdb.profile_system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.utils import volume_average
from fusdb.variable import Variable


def test_dt_reaction_rate_uses_supplied_volume_measure():
    rho = np.linspace(0.0, 1.0, 9)
    weight = 0.2 + rho**3
    n_D = 1.0 + rho
    n_T = 2.0 - 0.5 * rho
    sigmav = 3.0 + rho**2
    volume = 5.0
    relation = RELATIONS.get("DT reaction rate")

    result = relation.evaluate(
        {
            "n_D": n_D,
            "n_T": n_T,
            "sigmav_DT": sigmav,
            "V_p": volume,
            "rho": rho,
            "w_V": weight,
        }
    )
    expected = volume * volume_average(n_D * n_T * sigmav, rho, weight=weight)

    assert result == pytest.approx(expected)
    assert "w_V" in relation.constant_names


def test_dynamic_volume_measure_is_graph_dependency_for_nonprofile_physics():
    rho = np.linspace(0.0, 1.0, 9)
    weight = 0.2 + rho**3
    relation = RELATIONS.get("DT reaction rate")
    system = build_relation_system(
        [Variable("w_V", value=weight)],
        [relation],
        profile_size=rho.size,
    )
    migrated = next(
        item for item in system.candidate_primary_relations if item.name == relation.name
    )

    assert "w_V" in migrated.input_names
    assert "w_V" not in migrated.constant_names


def test_reduced_static_volume_measure_stays_off_physics_graph():
    relation = RELATIONS.get("DT reaction rate")
    fallback = RELATIONS.get("Reduced stellarator volume integration weight")
    system = build_relation_system([], [fallback, relation], profile_size=9)
    migrated = next(
        item for item in system.candidate_primary_relations if item.name == relation.name
    )

    assert "w_V" not in migrated.input_names
    assert "w_V" in migrated.constant_names


def test_thermal_pressure_uses_supplied_volume_measure():
    rho = np.linspace(0.0, 1.0, 9)
    weight = 0.2 + rho**2
    relation = RELATIONS.get("Thermal pressure")
    values = {
        "n_e": 1.0e20 * (1.0 + rho),
        "T_e": 5.0 + rho,
        "n_i": 0.8e20 * (1.0 + 0.5 * rho),
        "T_i": 4.0 + 2.0 * rho,
        "rho": rho,
        "w_V": weight,
    }

    weighted = relation.evaluate(values)
    legacy = relation.evaluate({key: value for key, value in values.items() if key != "w_V"})

    assert np.isfinite(weighted)
    assert weighted != pytest.approx(legacy)
    assert "w_V" in relation.constant_names
