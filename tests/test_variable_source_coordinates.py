import numpy as np
import pytest

from fusdb.variable import Variable


def test_profile_can_retain_explicit_source_coordinate_grid():
    source = np.linspace(0.0, 1.0, 101)
    values = 1.0e20 * (1.0 - 0.5 * source**2)

    var = Variable("n_e", value=values, coordinate="rho", coordinate_values=source)

    assert var.coordinate == "rho"
    assert var.size == 101
    assert var.has_source_grid
    assert np.array_equal(var.coordinate_values, source)
    assert np.array_equal(var.input_value, values)


def test_source_grid_length_overrides_nominal_profile_size():
    source = np.linspace(0.0, 1.0, 101)
    values = np.linspace(1.0e20, 5.0e19, 101)

    var = Variable("n_e", value=values, size=46, coordinate="rho", coordinate_values=source)

    assert var.size == 101


def test_profile_coordinate_values_default_to_common_rho_name():
    source = np.linspace(0.0, 1.0, 11)
    var = Variable("T_e", value=np.ones(11), coordinate_values=source)

    assert var.coordinate == "rho"


def test_profile_source_grid_must_match_profile_length():
    with pytest.raises(ValueError, match="does not match coordinate_values length"):
        Variable(
            "T_e",
            value=np.ones(10),
            coordinate="rho",
            coordinate_values=np.linspace(0.0, 1.0, 11),
        )


def test_profile_source_grid_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        Variable(
            "T_e",
            value=np.ones(3),
            coordinate="rho",
            coordinate_values=np.asarray([0.0, 0.8, 0.7]),
        )


def test_scalar_variable_rejects_source_coordinate():
    with pytest.raises(ValueError, match="Scalar variable"):
        Variable("V_p", value=100.0, coordinate="rho")
