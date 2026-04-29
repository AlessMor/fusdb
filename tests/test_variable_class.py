import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.variable_class import Variable


def test_variable_make_returns_variable_with_requested_ndim():
    """Expected: ``Variable.make`` always returns Variable and preserves ndim."""
    scalar = Variable.make(name="x", ndim=0)
    profile = Variable.make(name="profile", ndim=1)

    assert isinstance(scalar, Variable)
    assert isinstance(profile, Variable)
    assert scalar.ndim == 0
    assert profile.ndim == 1


def test_variable_rejects_invalid_ndim():
    """Expected: Variable only supports ndim=0 or ndim=1."""
    with pytest.raises(ValueError):
        Variable(name="bad", ndim=2)


def test_variable_loads_registry_constraints_by_default():
    """Expected: registry constraints live on Variable objects."""
    var = Variable.make(name="f_GW", ndim=0)
    assert var.constraints == ("f_GW >= 0",)


def test_variable_scalar_add_value_and_input_snapshot():
    """Expected: scalar writes update current/input values and reject invalid values."""
    var = Variable(name="x", ndim=0)

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


def test_variable_profile_add_value_from_scalar_and_array():
    """Expected: profile writes support scalar broadcast and arrays, and identical rewrites are no-ops."""
    var = Variable(name="n_i", ndim=1, profile_size=4)

    assert var.add_value(2.5, as_input=True) is True
    assert np.array_equal(var.current_value, np.full(4, 2.5))
    assert np.array_equal(var.input_value, np.full(4, 2.5))
    assert float(np.mean(var.current_value)) == pytest.approx(2.5)
    assert float(np.mean(var.input_value)) == pytest.approx(2.5)

    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    assert var.add_value(arr) is True
    assert np.array_equal(var.current_value, arr)
    assert float(np.mean(var.current_value)) == pytest.approx(2.5)
    assert var.add_value(arr.copy()) is False


def test_variable_profile_rejects_invalid_values():
    """Expected: malformed profile inputs are rejected while None writes are ignored."""
    var = Variable(name="T_i", ndim=1)

    assert var.add_value(None) is False

    with pytest.raises(ValueError):
        var.add_value(np.array([[1.0, 2.0]], dtype=float))

    with pytest.raises(ValueError):
        var.add_value(np.array([1.0, np.nan], dtype=float))

    with pytest.raises(ValueError):
        var.add_value("bad")
