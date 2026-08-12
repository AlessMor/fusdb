import numpy as np
import pytest

from fusdb.profile_system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.utils.profiles import volume_average
from fusdb.variable import Variable


PARABOLIC = "Parabolic electron density profile"
SAUTER_VOLUME = "Sauter self-similar profile volume mapping"


def test_legacy_rho_weight_keeps_parabolic_profile_exactly_unchanged():
    relation = RELATIONS.get(PARABOLIC)
    rho = np.linspace(0.0, 1.0, 46)
    namespace = {"n_e_avg": 4.2e19, "density_peaking": 1.7, "rho": rho}

    legacy = np.asarray(relation.evaluate(namespace), dtype=float)
    explicit = np.asarray(relation.evaluate({**namespace, "w_V": rho.copy()}), dtype=float)

    np.testing.assert_array_equal(explicit, legacy)


def test_nontrivial_volume_weight_controls_average_and_peaking_contract():
    relation = RELATIONS.get(PARABOLIC)
    rho = np.linspace(0.0, 1.0, 46)
    weight = rho * (1.0 - 0.18 * rho)
    average = 4.2e19
    peaking = 1.7

    profile = np.asarray(
        relation.evaluate(
            {
                "n_e_avg": average,
                "density_peaking": peaking,
                "rho": rho,
                "w_V": weight,
            }
        ),
        dtype=float,
    )

    assert volume_average(profile, rho, weight=weight) == pytest.approx(average, rel=2e-12)
    assert profile[0] / average == pytest.approx(peaking, rel=3e-4)


def test_batched_sauter_weights_normalize_each_profile_row():
    mapping = RELATIONS.get(SAUTER_VOLUME)
    profile_relation = RELATIONS.get(PARABOLIC)
    rho = np.linspace(0.0, 1.0, 46)
    delta = np.array([0.0, 0.25, 0.55])
    eps = np.array([0.30, 0.30, 0.30])
    _v_norm, weight = mapping.evaluate({"delta": delta, "eps": eps, "rho": rho})
    averages = np.array([3.0e19, 4.0e19, 5.0e19])
    peakings = np.array([1.3, 1.7, 2.1])

    profiles = np.asarray(
        profile_relation.evaluate(
            {
                "n_e_avg": averages,
                "density_peaking": peakings,
                "rho": rho,
                "w_V": weight,
            }
        ),
        dtype=float,
    )

    assert profiles.shape == (3, rho.size)
    np.testing.assert_allclose(
        volume_average(profiles, rho, weight=weight),
        averages,
        rtol=2e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(profiles[:, 0] / averages, peakings, rtol=3e-4, atol=0.0)


def test_geometry_change_recomputes_weighted_parabolic_profile_without_extra_dofs():
    mapping = RELATIONS.get(SAUTER_VOLUME)
    profile_relation = RELATIONS.get(PARABOLIC)
    system = build_relation_system(
        [
            Variable("delta", value=0.15),
            Variable("eps", value=0.30),
            Variable("n_e_avg", value=4.0e19),
            Variable("density_peaking", value=1.8),
        ],
        (mapping, profile_relation),
        profile_size=46,
    )
    system.compile()
    system.pack()

    base = system.complete(system.solver_values())
    profile0 = np.asarray(base["n_e"], dtype=float).copy()
    weight0 = np.asarray(base["w_V"], dtype=float).copy()

    changed = dict(base)
    changed["delta"] = 0.55
    system.complete(changed)
    profile1 = np.asarray(changed["n_e"], dtype=float)
    weight1 = np.asarray(changed["w_V"], dtype=float)
    rho = np.asarray(changed["rho"], dtype=float)

    assert not np.allclose(weight0, weight1)
    assert not np.allclose(profile0, profile1)
    assert volume_average(profile1, rho, weight=weight1) == pytest.approx(4.0e19, rel=2e-12)
    assert "w_V" not in system.packed_variables
    assert "n_e" not in system.packed_variables
