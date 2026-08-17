import numpy as np

from fusdb.profile_system import build_relation_system
from fusdb.relation import Relation
from fusdb.variable import Variable


def test_supplied_physical_coordinate_mapping_is_held_not_packed():
    grid = np.linspace(0.0, 1.0, 46) ** 1.2
    relation = Relation(
        name="Synthetic toroidal-coordinate consumer",
        func=lambda mapping: float(np.mean(mapping)),
        input_names=("rho_tor",),
        outputs=("n_la",),
        argument_names=("mapping",),
        function_name="synthetic_toroidal_coordinate_consumer",
    )
    system = build_relation_system(
        [Variable("rho_tor", value=grid)],
        [relation],
        profile_size=46,
    )
    system.compile()
    system.pack()

    assert "rho_tor" in system.fixed
    assert "rho_tor" not in system.packed_variables
    assert all(spec[0] != "rho_tor" for spec in system.packed_specs)
    np.testing.assert_allclose(system.inputs["rho_tor"], grid)


def test_derived_tokamak_mapping_is_computed_not_packed():
    rho_relation = Relation(
        name="Synthetic minor-radius mapping",
        func=lambda rho: np.asarray(rho, dtype=float),
        input_names=(),
        outputs=("rho_minor",),
        constant_names=("rho",),
        function_name="synthetic_minor_radius_mapping",
    )
    consumer = Relation(
        name="Synthetic minor-radius consumer",
        func=lambda mapping: float(np.mean(mapping)),
        input_names=("rho_minor",),
        outputs=("n_la",),
        argument_names=("mapping",),
        function_name="synthetic_minor_radius_consumer",
    )
    system = build_relation_system([], [rho_relation, consumer], profile_size=46)
    system.compile()
    system.pack()

    assert "rho_minor" not in system.packed_variables
    assert all(spec[0] != "rho_minor" for spec in system.packed_specs)
