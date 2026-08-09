import numpy as np
import pytest

from fusdb.utils.profiles import coordinate_average, normalized_shape, trapezoid, volume_average


def test_volume_average_without_geometry_mapping_preserves_legacy_weighting():
    rho = np.linspace(0.0, 1.0, 46)
    profile = 2.0 + 3.0 * rho**2

    expected = trapezoid(profile * rho, x=rho) / trapezoid(rho, x=rho)

    assert volume_average(profile, rho) == pytest.approx(expected, rel=0.0, abs=0.0)


def test_volume_average_can_use_geometry_supplied_normalized_volume():
    rho = np.linspace(0.0, 1.0, 31)
    v_norm = rho**3

    # A function linear in enclosed normalized volume has exact mean 1/2 under
    # trapezoidal integration when integrated against that same coordinate.
    assert volume_average(v_norm, rho, v_norm=v_norm) == pytest.approx(0.5)


def test_normalized_shape_has_unit_volume_average_on_selected_mapping():
    rho = np.linspace(0.0, 1.0, 51)
    v_norm = rho**2.4
    profile = 4.0 * (1.0 - 0.8 * rho**2) + 0.2

    avg, shape = normalized_shape(profile, rho, v_norm=v_norm)

    assert avg == pytest.approx(volume_average(profile, rho, v_norm=v_norm))
    assert volume_average(shape, rho, v_norm=v_norm) == pytest.approx(1.0)
    assert np.allclose(np.asarray(avg) * shape, profile)


def test_coordinate_average_requires_strictly_monotonic_mapping():
    profile = np.asarray([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        coordinate_average(profile, np.asarray([0.0, 0.8, 0.7]))
