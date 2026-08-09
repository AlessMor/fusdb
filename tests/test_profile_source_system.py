import numpy as np
import pytest

from fusdb.profile_sources import prepare_source_profiles
from fusdb.relationsystem import RelationSystem
from fusdb.utils.profiles import volume_average
from fusdb.variable import Variable


def _source_system(*, fixed_profile=False):
    common_size = 46
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_profile = 2.0 + 3.0 * source_coordinate**2
    mapping = np.linspace(0.0, 1.0, common_size)

    # B is used here only as a registered profile-shaped mapping variable so the
    # test exercises the normal RelationSystem graph without depending on the
    # later rho_tor registry migration.  The source adapter treats coordinate
    # names generically; production mappings are rho_minor/rho_tor/etc.
    variables = [
        Variable("B", value=mapping, fixed=True, size=common_size),
        Variable(
            "n_e",
            value=source_profile,
            coordinate="B",
            coordinate_values=source_coordinate,
            fixed=fixed_profile,
        ),
    ]
    prepared, relations, size = prepare_source_profiles(
        variables, (), profile_size=common_size
    )
    system = RelationSystem(prepared, relations, name="source_profile_test")
    system.compile()
    return system, source_profile, source_coordinate, size


def test_source_profile_grid_does_not_set_relation_system_grid_size():
    system, _source_profile, _source_coordinate, size = _source_system()

    assert size == 46
    assert system.profile_size == 46
    assert system.values["B"].shape == (46,)


def test_movable_source_profile_is_one_scalar_amplitude_not_pointwise_dofs():
    system, _source_profile, _source_coordinate, _size = _source_system()
    system.pack()

    spans = {name: stop - start for name, start, stop, *_rest in system.packed_specs}
    assert "n_e" not in spans
    assert spans.get("n_e_avg") == 1


def test_source_profile_remaps_when_geometry_mapping_changes():
    system, _source_profile, _source_coordinate, _size = _source_system()
    base = system.complete(system.solver_values())
    profile_1 = np.asarray(base["n_e"], dtype=float).copy()
    average = float(np.asarray(base["n_e_avg"]).reshape(-1)[0])

    changed = dict(base)
    changed["B"] = np.linspace(0.0, 1.0, system.profile_size) ** 1.5
    system.complete(changed)
    profile_2 = np.asarray(changed["n_e"], dtype=float)

    assert not np.allclose(profile_1, profile_2)
    rho = np.asarray(changed["rho"], dtype=float)
    assert volume_average(profile_2, rho) == pytest.approx(average)


def test_fixed_source_profile_keeps_absolute_source_values_while_remapping():
    system, source_profile, source_coordinate, _size = _source_system(fixed_profile=True)
    base = system.complete(system.solver_values())
    profile_1 = np.asarray(base["n_e"], dtype=float).copy()

    changed = dict(base)
    mapping = np.linspace(0.0, 1.0, system.profile_size) ** 1.5
    changed["B"] = mapping
    system.complete(changed)
    profile_2 = np.asarray(changed["n_e"], dtype=float)

    expected = np.interp(mapping, source_coordinate, source_profile)
    assert not np.allclose(profile_1, profile_2)
    assert np.allclose(profile_2, expected)


def test_source_profile_conversion_fails_cleanly_when_mapping_exceeds_coverage():
    system, _source_profile, _source_coordinate, _size = _source_system()
    values = system.solver_values()
    values["B"] = np.linspace(0.0, 1.01, system.profile_size)

    system.complete(values)

    error = system.completion_errors.get("Source profile n_e on B", "")
    assert "outside source coverage" in error
