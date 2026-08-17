"""Profile <-> volume-average consistency is enforced by a real relation.

A supplied profile and its scalar average are linked by an outputless
``<x>_avg == volume_average(<x>)`` residual rather than a build-time override:
if a fixed profile and a separately supplied average disagree, the mode surfaces
it (verify) or reconciles it (reconcile), instead of silently overwriting the
average and warning at construction time.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusdb.utils import line_average, volume_average
from fusdb.registry import KEV_TO_J, RELATIONS, VARIABLES
from fusdb.relationsystem import RelationSystem
from fusdb.variable import Variable

_CONSISTENCY = "Electron temperature volume-average consistency"


def _system(t_e_avg: float, profile: np.ndarray, *, avg_fixed: bool = False) -> RelationSystem:
    """A minimal system: a fixed T_e profile plus a supplied scalar T_e_avg."""
    rho = np.linspace(0.0, 1.0, profile.size)
    rel = RELATIONS.get(_CONSISTENCY)
    variables = [
        Variable("rho", value=rho, fixed=True),
        Variable("w_V", value=rho, fixed=True),
        Variable("T_e", value=profile, fixed=True),
        Variable("T_e_avg", value=t_e_avg, fixed=avg_fixed),
    ]
    return RelationSystem(variables, [rel], name="profile_average_test").compile()


def test_fixed_profile_conflicting_average_is_flagged_on_verify():
    profile = np.full(21, 15.0)
    system = _system(14.0, profile)
    result = system.run("verify")
    assert not result["success"]
    assert not result["relation_status"][_CONSISTENCY]["verified"]


def test_fixed_profile_consistent_average_passes_verify():
    profile = np.full(21, 15.0)
    system = _system(15.0, profile)
    result = system.run("verify")
    assert result["relation_status"][_CONSISTENCY]["verified"]


def test_reconcile_moves_supplied_average_to_the_profile_value():
    profile = np.full(21, 15.0)
    system = _system(14.0, profile)
    result = system.run("reconcile")
    assert result["success"]
    assert system.values["T_e_avg"] == pytest.approx(15.0, abs=1e-3)


def test_profile_avg_uses_volume_average_not_line_average():
    rho = np.linspace(0.0, 1.0, 101)
    profile = rho.copy()
    expected_volume = volume_average(profile, rho)
    expected_rho = line_average(profile, rho)
    assert expected_volume != pytest.approx(expected_rho)

    system = _system(float(expected_volume), profile)
    result = system.run("verify")
    assert result["relation_status"][_CONSISTENCY]["verified"]


def test_explicit_rho_average_relation_uses_straight_line_average():
    rho = np.linspace(0.0, 1.0, 101)
    profile = rho.copy()
    rel = RELATIONS.get("Electron temperature rho-average")
    result = rel.evaluate({"T_e": profile, "rho": rho})
    assert result == pytest.approx(line_average(profile, rho))


def test_line_average_uses_normalized_minor_radius_definition():
    rho_minor = np.linspace(0.0, 1.0, 101)
    profile = rho_minor**2
    assert line_average(profile, rho_minor) == pytest.approx(1.0 / 3.0, rel=2e-4)


def test_electron_density_profile_produces_conventional_line_average():
    rho_minor = np.linspace(0.0, 1.0, 101)
    profile = 2.0 - rho_minor
    rel = RELATIONS.get("Electron density line-average")
    assert rel.evaluate({"n_e": profile, "rho_minor": rho_minor}) == pytest.approx(1.5)
    assert VARIABLES.resolve("n_e_la") == "n_la"


def test_profile_line_average_provides_n_la():
    rho = np.linspace(0.0, 1.0, 101)
    profile = 2.0 - rho
    variables = [
        Variable("rho", value=rho, fixed=True),
        Variable("rho_minor", value=rho, fixed=True),
        Variable("w_V", value=rho, fixed=True),
        Variable("n_e", value=profile, fixed=True),
    ]
    relations = [
        RELATIONS.get("Electron density line-average"),
        RELATIONS.get("Electron density volume-average consistency"),
    ]
    system = RelationSystem(variables, relations, name="line_average_provider_test").compile()
    result = system.run("verify")
    assert result["success"]
    system.complete(system.values)
    assert system.values["n_la"] == pytest.approx(1.5)


def test_thermal_pressure_has_volume_and_rho_averaged_outputs():
    rho = np.linspace(0.0, 1.0, 101)
    n_e = np.ones_like(rho)
    T_e = rho.copy()
    n_i = np.zeros_like(rho)
    T_i = np.zeros_like(rho)

    volume_rel = RELATIONS.get("Thermal pressure")
    rho_rel = RELATIONS.get("Thermal pressure rho-average")

    values = {"n_e": n_e, "T_e": T_e, "n_i": n_i, "T_i": T_i, "rho": rho}
    assert volume_rel.evaluate(values) == pytest.approx(KEV_TO_J * volume_average(T_e, rho))
    assert rho_rel.evaluate(values) == pytest.approx(KEV_TO_J * line_average(T_e, rho))


def test_stored_energy_uses_volume_averaged_pressure():
    rel = RELATIONS.get("Thermal stored energy")
    assert rel.evaluate({"p_th": 4.0, "V_p": 10.0}) == pytest.approx(60.0)


def test_shape_locked_profile_residual_is_trivially_satisfied():
    """A level-free profile reconstructed as avg*shape never fights the residual."""
    rho = np.linspace(0.0, 1.0, 51)
    shape = (1.0 - rho**2) ** 1.5 + 0.1
    profile = 8.0 * shape
    variables = [
        Variable("rho", value=rho, fixed=True),
        Variable("w_V", value=rho, fixed=True),
        Variable("T_e", value=profile, fixed=False),
    ]
    system = RelationSystem(variables, [RELATIONS.get(_CONSISTENCY)], name="level_free_test").compile()
    result = system.run("verify")
    assert result["relation_status"][_CONSISTENCY]["verified"]
