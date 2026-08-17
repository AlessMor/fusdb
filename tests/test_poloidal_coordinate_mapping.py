import numpy as np

from fusdb.profile_system import build_relation_system
from fusdb.registry import VARIABLES
from fusdb.variable import Variable


def test_poloidal_flux_coordinate_is_registered_without_a_reduced_default():
    spec = VARIABLES.get("rho_pol")

    assert spec.shape == 1
    assert spec.default_relation == ()
    assert VARIABLES.resolve("normalized_poloidal_flux_radius") == "rho_pol"
    assert VARIABLES.resolve("rho_poloidal") == "rho_pol"


def test_supplied_poloidal_flux_mapping_can_coordinate_a_source_profile():
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_values = 3.0e19 * (1.0 - 0.4 * source_coordinate**2)
    target_mapping = np.linspace(0.0, 1.0, 31) ** 1.15

    system = build_relation_system(
        [
            Variable("rho_pol", value=target_mapping),
            Variable(
                "n_e",
                value=source_values,
                coordinate="rho_pol",
                coordinate_values=source_coordinate,
            ),
        ],
        (),
        profile_size=31,
    )
    system.compile()
    system.pack()
    completed = system.complete(system.solver_values())

    assert system.profile_size == 31
    assert "rho_pol" not in system.packed_variables
    assert "n_e" not in system.packed_variables
    np.testing.assert_array_equal(completed["rho_pol"], target_mapping)
    assert np.asarray(completed["n_e"]).shape == (31,)
