import numpy as np
import pytest

from fusdb.registry import RELATIONS
from fusdb.profiles.numerics import volume_average


def test_process_core_radiation_declares_minor_radius_and_volume_weight():
    relation = RELATIONS.get("Core radiation power (PROCESS)")

    assert "rho_minor" in relation.input_names
    assert "w_V" in relation.input_names
    assert "rho" not in relation.input_names
    assert "rho" in relation.constant_names


def test_process_core_radiation_identity_mapping_preserves_legacy_fraction():
    rho = np.linspace(0.0, 1.0, 101)
    n_e = 8.0e19 * (1.0 - 0.4 * rho**2) + 1.0e19
    T_e = 10.0 * (1.0 - 0.6 * rho**2) + 1.0
    P_brem = 40.0e6
    P_sync = 5.0e6
    core_radius = 0.75
    reduction = 0.6

    relation = RELATIONS.get("Core radiation power (PROCESS)")
    value = relation.evaluate(
        {
            "n_e": n_e,
            "T_e": T_e,
            "rho_minor": rho,
            "w_V": rho,
            "rho": rho,
            "P_cool_imp": 0.0,
            "P_brem": P_brem,
            "P_sync": P_sync,
            "radius_plasma_core_norm": core_radius,
            "f_p_plasma_core_rad_reduction": reduction,
        }
    )

    hydrogenic_shape = (n_e / 1.0e20) ** 2 * np.sqrt(T_e)
    inside = (rho <= core_radius).astype(float)
    fraction = volume_average(hydrogenic_shape * inside, rho) / volume_average(hydrogenic_shape, rho)
    expected = reduction * P_brem * fraction + P_sync

    assert value == pytest.approx(expected, rel=2.0e-14)
