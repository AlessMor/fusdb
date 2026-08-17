from pathlib import Path

import numpy as np

from fusdb import Reactor


GIGA = Path(__file__).parents[1] / "reactors" / "GIGA" / "reactor.yaml"


def test_giga_uses_explicit_source_coordinates_with_independent_sample_counts():
    reactor = Reactor.from_yaml(GIGA)

    assert reactor.variables["T_e"].coordinate == "rho_minor"
    assert reactor.variables["n_e"].coordinate == "rho_minor"
    assert reactor.variables["T_e"].size != reactor.variables["n_e"].size
    assert reactor.variables["T_e"].coordinate_values[0] == 0.0
    assert reactor.variables["T_e"].coordinate_values[-1] == 1.0
    assert reactor.variables["n_e"].coordinate_values[0] == 0.0
    assert reactor.variables["n_e"].coordinate_values[-1] == 1.0

    # The source sample counts no longer compete to define the solver grid.
    system = reactor.relation_system()
    assert system.profile_size == 46
    system.compile()
    values = system.complete(system.solver_values())

    assert np.asarray(values["rho_minor"]).shape == (46,)
    assert np.asarray(values["T_e"]).shape == (46,)
    assert np.asarray(values["n_e"]).shape == (46,)
    assert "rho_minor" not in system.packed_variables


def test_giga_endpoint_extension_is_explicit_not_interpolator_extrapolation():
    reactor = Reactor.from_yaml(GIGA)
    for name in ("T_e", "n_e"):
        variable = reactor.variables[name]
        source = np.asarray(variable.coordinate_values, dtype=float)
        values = np.asarray(variable.input_value, dtype=float)
        assert source[0] == 0.0 and source[-1] == 1.0
        # The explicit zero-order endpoint model is encoded in the source data,
        # while reinterpolate_profile itself remains strict.
        assert values[0] == values[1]
        assert values[-1] == values[-2]
