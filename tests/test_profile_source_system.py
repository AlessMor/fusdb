import pickle

import numpy as np
import pytest

from fusdb.profile_sources import prepare_source_profiles
from fusdb.relation import Relation
from fusdb.relationsystem import RelationSystem
from fusdb.utils.profiles import volume_average
from fusdb.variable import Variable


def _mapping_relation(*, scaled=False):
    if scaled:
        def mapping(A, *, rho):
            return np.asarray(rho, dtype=float) * (A / 2.0)
    else:
        def mapping(A, *, rho):
            return np.asarray(rho, dtype=float) ** (A / 2.0)
    return Relation(
        name="Synthetic coordinate mapping" + (" scaled" if scaled else ""),
        func=mapping,
        input_names=("A",),
        outputs=("B",),
        constant_names=("rho",),
        function_name="synthetic_coordinate_mapping" + ("_scaled" if scaled else ""),
        argument_names=("A",),
        source_kind="test",
        source_name="test",
    )


def _source_system(*, fixed_profile=False, dynamic_mapping=True, scaled_mapping=False):
    common_size = 46
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_profile = 2.0 + 3.0 * source_coordinate**2

    if dynamic_mapping:
        variables = [
            Variable("A", value=2.0),
            Variable(
                "n_e",
                value=source_profile,
                coordinate="B",
                coordinate_values=source_coordinate,
                fixed=fixed_profile,
            ),
        ]
        relations = (_mapping_relation(scaled=scaled_mapping),)
    else:
        # Direct supplied mapping exercises common-grid selection only. A fixed
        # ordinary profile is intentionally immutable under completion, so
        # dynamic remapping tests use the derived mapping branch above.
        variables = [
            Variable("B", value=np.linspace(0.0, 1.0, common_size), fixed=True, size=common_size),
            Variable(
                "n_e",
                value=source_profile,
                coordinate="B",
                coordinate_values=source_coordinate,
                fixed=fixed_profile,
            ),
        ]
        relations = ()

    prepared, relations, size = prepare_source_profiles(
        variables, relations, profile_size=common_size
    )
    system = RelationSystem(prepared, relations, name="source_profile_test").compile()
    return system, source_profile, source_coordinate, size


def test_source_profile_grid_does_not_set_relation_system_grid_size():
    system, _source_profile, _source_coordinate, size = _source_system(dynamic_mapping=False)

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
    changed["A"] = 1.5
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
    changed["A"] = 1.5
    system.complete(changed)
    profile_2 = np.asarray(changed["n_e"], dtype=float)

    mapping = np.asarray(changed["B"], dtype=float)
    expected = np.interp(mapping, source_coordinate, source_profile)
    assert not np.allclose(profile_1, profile_2)
    assert np.allclose(profile_2, expected)


def test_generated_source_relation_is_picklable():
    system, _source_profile, _source_coordinate, _size = _source_system()
    relation = next(rel for rel in system.model.candidate_primary_relations if rel.source_kind == "source_profile")

    restored = pickle.loads(pickle.dumps(relation, protocol=pickle.HIGHEST_PROTOCOL))
    values = system.complete(system.solver_values())

    expected = relation.output_map(relation.evaluate(values))["n_e"]
    actual = restored.output_map(restored.evaluate(values))["n_e"]
    assert np.allclose(actual, expected)


def test_source_profile_conversion_fails_cleanly_when_mapping_exceeds_coverage():
    system, _source_profile, _source_coordinate, _size = _source_system(scaled_mapping=True)
    values = system.complete(system.solver_values())
    values["A"] = 2.02

    system.complete(values)

    error = system.completion_errors.get("Source profile n_e on B", "")
    assert "outside source coverage" in error
