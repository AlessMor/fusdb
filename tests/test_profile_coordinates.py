import numpy as np
import pytest

from fusdb.utils.profiles import coordinate_average, normalized_shape, trapezoid, volume_average, weighted_average


def test_volume_average_without_geometry_mapping_preserves_legacy_weighting():
    rho = np.linspace(0.0, 1.0, 46)
    profile = 2.0 + 3.0 * rho**2

    expected = trapezoid(profile * rho, x=rho) / trapezoid(rho, x=rho)

    assert volume_average(profile, rho) == pytest.approx(expected, rel=0.0, abs=0.0)


def test_default_volume_weight_reproduces_legacy_formula_exactly():
    rho = np.linspace(0.0, 1.0, 46)
    profile = 1.3 + 0.7 * rho + 2.1 * rho**3

    assert volume_average(profile, rho, weight=rho) == pytest.approx(
        volume_average(profile, rho), rel=0.0, abs=0.0
    )


def test_weight_normalization_is_irrelevant():
    rho = np.linspace(0.0, 1.0, 46)
    profile = 2.0 - 0.4 * rho**2

    assert weighted_average(profile, rho, 7.0 * rho) == pytest.approx(
        weighted_average(profile, rho, rho)
    )


def test_volume_average_can_use_geometry_supplied_normalized_volume():
    rho = np.linspace(0.0, 1.0, 31)
    v_norm = rho**3

    # A function linear in enclosed normalized volume has exact mean 1/2 under
    # trapezoidal integration when integrated against that same coordinate.
    assert volume_average(v_norm, rho, v_norm=v_norm) == pytest.approx(0.5)


def test_volume_average_rejects_ambiguous_geometry_measure():
    rho = np.linspace(0.0, 1.0, 11)

    with pytest.raises(ValueError, match="either weight or v_norm"):
        volume_average(rho, rho, weight=rho, v_norm=rho**2)


def test_normalized_shape_has_unit_volume_average_on_selected_mapping():
    rho = np.linspace(0.0, 1.0, 51)
    v_norm = rho**2.4
    profile = 4.0 * (1.0 - 0.8 * rho**2) + 0.2

    avg, shape = normalized_shape(profile, rho, v_norm=v_norm)

    assert avg == pytest.approx(volume_average(profile, rho, v_norm=v_norm))
    assert volume_average(shape, rho, v_norm=v_norm) == pytest.approx(1.0)
    assert np.allclose(np.asarray(avg) * shape, profile)


def test_normalized_shape_supports_geometry_weight():
    rho = np.linspace(0.0, 1.0, 51)
    weight = rho * (1.0 + 0.2 * rho)
    profile = 3.0 * (1.0 - 0.5 * rho**2) + 0.1

    avg, shape = normalized_shape(profile, rho, weight=weight)

    assert volume_average(shape, rho, weight=weight) == pytest.approx(1.0)
    assert np.allclose(np.asarray(avg) * shape, profile)


def test_coordinate_average_requires_strictly_monotonic_mapping():
    profile = np.asarray([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        coordinate_average(profile, np.asarray([0.0, 0.8, 0.7]))


def test_weighted_average_rejects_negative_weights():
    profile = np.asarray([1.0, 2.0, 3.0])
    rho = np.asarray([0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        weighted_average(profile, rho, np.asarray([0.0, -1.0, 1.0]))
