import numpy as np

from fusdb.registry import RELATIONS


def _bootstrap_inputs():
    rho_minor = np.linspace(0.0, 1.0, 46)
    # Keep profiles finite at the separatrix; this test is about the coordinate
    # contract, not edge singularities of a particular profile model.
    return {
        "n_e": 8.0e19 * (1.0 - 0.5 * rho_minor**2) + 1.0e19,
        "T_e": 12.0 * (1.0 - 0.7 * rho_minor**2) + 1.0,
        "n_i": 7.5e19 * (1.0 - 0.45 * rho_minor**2) + 1.0e19,
        "T_i": 13.0 * (1.0 - 0.65 * rho_minor**2) + 1.0,
        "rho_minor": rho_minor,
        "S_phi": 20.0,
        "R": 6.2,
        "a": 2.0,
        "B0": 5.3,
        "delta": 0.3,
        "q0": 1.0,
        "q95": 3.2,
        "Z_eff": 1.8,
        "I_p": 15.0e6,
        "afuel": 2.5,
    }


def test_sauter_bootstrap_declares_physical_minor_radius_dependency():
    relation = RELATIONS.get("Bootstrap fraction Sauter")

    assert "rho_minor" in relation.input_names
    assert "rho" not in relation.input_names
    assert "rho" not in relation.constant_names


def test_sauter_bootstrap_runs_with_identity_tokamak_minor_radius_mapping():
    relation = RELATIONS.get("Bootstrap fraction Sauter")
    value = relation.func(**_bootstrap_inputs())

    assert np.isfinite(value)
