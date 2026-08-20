import numpy as np
import pytest

from fusdb.profiles.sources import prepare_source_profiles
from fusdb.registry import RELATIONS, TAGS
from fusdb.relationsystem import RelationSystem
from fusdb.profiles.numerics import volume_average
from fusdb.variable import Variable


NAME = "Sauter self-similar profile volume mapping"


def test_sauter_profile_volume_mapping_recovers_legacy_circular_limit_exactly():
    relation = RELATIONS.get(NAME)
    rho = np.linspace(0.0, 1.0, 101)
    v_norm, w_V = relation.evaluate({"delta": 0.0, "eps": 0.35, "rho": rho})

    np.testing.assert_array_equal(v_norm, rho**2)
    np.testing.assert_array_equal(w_V, rho)


def test_sauter_profile_volume_mapping_responds_to_geometry():
    relation = RELATIONS.get(NAME)
    rho = np.linspace(0.0, 1.0, 101)
    v0, w0 = relation.evaluate({"delta": 0.0, "eps": 0.3, "rho": rho})
    v1, w1 = relation.evaluate({"delta": 0.5, "eps": 0.3, "rho": rho})

    assert not np.allclose(v0, v1)
    assert not np.allclose(w0, w1)
    assert v1[0] == pytest.approx(0.0)
    assert v1[-1] == pytest.approx(1.0)
    assert np.all(np.diff(v1) > 0.0)
    assert np.all(w1 >= 0.0)

    # w_V is deliberately one half of d(v_norm)/d(rho): its overall scale is
    # irrelevant to a weighted average, and this convention recovers w_V=rho
    # exactly at zero triangularity.
    derivative = np.gradient(v1, rho, edge_order=2)
    np.testing.assert_allclose(2.0 * w1, derivative, rtol=3e-4, atol=3e-4)


def test_sauter_profile_volume_mapping_supports_batched_geometry_inputs():
    relation = RELATIONS.get(NAME)
    rho = np.linspace(0.0, 1.0, 51)
    delta = np.asarray([[0.0], [0.5]])
    eps = np.asarray([[0.3], [0.3]])

    v_norm, w_V = relation.evaluate({"delta": delta, "eps": eps, "rho": rho})

    assert v_norm.shape == (2, rho.size)
    assert w_V.shape == (2, rho.size)
    np.testing.assert_array_equal(v_norm[0], rho**2)
    np.testing.assert_array_equal(w_V[0], rho)
    assert not np.allclose(v_norm[0], v_norm[1])
    assert not np.allclose(w_V[0], w_V[1])
    np.testing.assert_allclose(v_norm[:, -1], 1.0)


def test_sauter_mapping_is_opt_in_and_atomic_for_both_volume_outputs():
    ordinary = {
        relation.name
        for relation in RELATIONS.get_filtered_relations(tags=TAGS.expand(("tokamak",)))
    }
    assert NAME not in ordinary
    assert "Tokamak normalized enclosed volume" in ordinary
    assert "Tokamak volume integration weight" in ordinary

    selected = {
        relation.name
        for relation in RELATIONS.get_filtered_relations(
            tags=TAGS.expand(("tokamak",)),
            default_relations={"v_norm": (NAME,)},
        )
    }
    assert NAME in selected
    assert "Tokamak normalized enclosed volume" not in selected
    assert "Tokamak volume integration weight" not in selected


def test_sauter_geometry_flows_through_mapping_and_source_profile_normalization():
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_profile = 2.0 + 4.0 * source_coordinate**2
    mapping = RELATIONS.get(NAME)
    variables = [
        Variable("delta", value=0.2),
        Variable("eps", value=0.3),
        Variable(
            "n_e",
            value=source_profile,
            coordinate="v_norm",
            coordinate_values=source_coordinate,
        ),
    ]
    prepared, relations, _size = prepare_source_profiles(
        variables,
        (mapping,),
        profile_size=46,
    )
    system = RelationSystem(prepared, relations, name="sauter_profile_mapping_test").compile()
    system.pack()

    base = system.complete(system.solver_values())
    profile0 = np.asarray(base["n_e"], dtype=float).copy()
    v0 = np.asarray(base["v_norm"], dtype=float).copy()
    average = float(base["n_e_avg"])

    changed = dict(base)
    changed["delta"] = 0.5
    system.complete(changed)
    profile1 = np.asarray(changed["n_e"], dtype=float)
    v1 = np.asarray(changed["v_norm"], dtype=float)
    w1 = np.asarray(changed["w_V"], dtype=float)
    rho = np.asarray(changed["rho"], dtype=float)

    assert not np.allclose(v0, v1)
    assert not np.allclose(profile0, profile1)
    assert volume_average(profile1, rho, weight=w1) == pytest.approx(average)
    assert "v_norm" not in system.packed_variables
    assert "w_V" not in system.packed_variables
