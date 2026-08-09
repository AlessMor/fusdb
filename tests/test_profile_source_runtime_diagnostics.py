import numpy as np

from fusdb.utils.profiles import volume_average
from tests.test_profile_source_system import _source_system


def _changed_movable():
    system, *_ = _source_system()
    base = system.complete(system.solver_values())
    profile_1 = np.asarray(base["n_e"], dtype=float).copy()
    average = float(np.asarray(base["n_e_avg"]).reshape(-1)[0])
    changed = dict(base)
    mapping = np.linspace(0.0, 1.0, system.profile_size) ** 1.5
    changed["B"] = mapping
    system.complete(changed)
    return system, base, changed, profile_1, average, mapping


def test_runtime_provider_executes_without_error():
    system, *_ = _changed_movable()
    assert not system.completion_errors


def test_runtime_mapping_is_not_overwritten():
    _system, _base, changed, _profile_1, _average, mapping = _changed_movable()
    assert np.allclose(changed["B"], mapping)


def test_runtime_profile_is_recomputed():
    _system, _base, changed, profile_1, _average, _mapping = _changed_movable()
    assert not np.allclose(profile_1, np.asarray(changed["n_e"], dtype=float))


def test_runtime_profile_preserves_average():
    system, _base, changed, _profile_1, average, _mapping = _changed_movable()
    rho = np.asarray(changed["rho"], dtype=float)
    assert np.isclose(volume_average(changed["n_e"], rho), average, rtol=1e-10, atol=0.0)


def _changed_fixed():
    system, source_profile, source_coordinate, _ = _source_system(fixed_profile=True)
    base = system.complete(system.solver_values())
    profile_1 = np.asarray(base["n_e"], dtype=float).copy()
    changed = dict(base)
    mapping = np.linspace(0.0, 1.0, system.profile_size) ** 1.5
    changed["B"] = mapping
    system.complete(changed)
    expected = np.interp(mapping, source_coordinate, source_profile)
    return system, changed, profile_1, expected


def test_runtime_fixed_provider_executes_without_error():
    system, *_ = _changed_fixed()
    assert not system.completion_errors


def test_runtime_fixed_profile_is_recomputed():
    _system, changed, profile_1, _expected = _changed_fixed()
    assert not np.allclose(profile_1, np.asarray(changed["n_e"], dtype=float))


def test_runtime_fixed_profile_matches_absolute_source():
    _system, changed, _profile_1, expected = _changed_fixed()
    assert np.allclose(changed["n_e"], expected)


def test_runtime_out_of_coverage_records_completion_error():
    system, *_ = _source_system()
    values = system.solver_values()
    values["B"] = np.linspace(0.0, 1.01, system.profile_size)
    system.complete(values)
    assert bool(system.completion_errors)


def test_runtime_out_of_coverage_error_is_source_relation():
    system, *_ = _source_system()
    values = system.solver_values()
    values["B"] = np.linspace(0.0, 1.01, system.profile_size)
    system.complete(values)
    assert "Source profile n_e on B" in system.completion_errors
