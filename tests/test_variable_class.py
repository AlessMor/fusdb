import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.variable_class import Variable, Variable0D, Variable1D
from fusdb.variable_util import make_variable


def test_variable_is_abstract():
    """Expected: instantiating the abstract Variable base class raises TypeError."""
    with pytest.raises(TypeError):
        Variable(name="x")


def test_make_variable_returns_expected_subclass():
    """Expected: make_variable dispatches to Variable0D for ndim=0 and Variable1D for ndim=1."""
    assert isinstance(make_variable(name="x", ndim=0), Variable0D)
    assert isinstance(make_variable(name="profile", ndim=1), Variable1D)


def test_variable0d_add_value_and_input_snapshot():
    """Expected: scalar writes update current/input values correctly, no-op on identical writes, and reject invalid values."""
    var = Variable0D(name="x")

    assert var.add_value(None) is False
    assert var.add_value(3.0, as_input=True) is True
    assert var.current_value == pytest.approx(3.0)
    assert var.input_value == pytest.approx(3.0)

    assert var.add_value(3.0) is False
    assert var.add_value(4.0) is True
    assert var.current_value == pytest.approx(4.0)
    assert var.input_value == pytest.approx(3.0)

    with pytest.raises(ValueError):
        var.add_value(float("nan"))

    with pytest.raises(ValueError):
        var.add_value("bad")


def test_variable1d_add_value_from_scalar_and_array():
    """Expected: profile writes support scalar broadcast and arrays, and identical array rewrites are no-ops."""
    var = Variable1D(name="n_i", profile_size=4)

    assert var.add_value(2.5, as_input=True) is True
    assert np.array_equal(var.current_value, np.full(4, 2.5))
    assert np.array_equal(var.input_value, np.full(4, 2.5))
    assert var.current_value_mean == pytest.approx(2.5)
    assert var.input_value_mean == pytest.approx(2.5)

    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    assert var.add_value(arr) is True
    assert np.array_equal(var.current_value, arr)
    assert var.current_value_mean == pytest.approx(2.5)
    assert var.add_value(arr.copy()) is False


def test_variable1d_rejects_invalid_values():
    """Expected: malformed profile inputs are rejected while None writes are ignored."""
    var = Variable1D(name="T_i")

    assert var.add_value(None) is False

    with pytest.raises(ValueError):
        var.add_value(np.array([[1.0, 2.0]], dtype=float))

    with pytest.raises(ValueError):
        var.add_value(np.array([1.0, np.nan], dtype=float))

    with pytest.raises(ValueError):
        var.add_value("bad")
