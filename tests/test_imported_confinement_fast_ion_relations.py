"""Contracts for the 2026-08 confinement/fast-ion source imports."""

from __future__ import annotations

import numpy as np
import pytest

from fusdb.reactor import Reactor
from fusdb.registry import KEV_TO_J, MU0, RELATIONS, VARIABLES
from fusdb.utils import volume_average
from fusdb.variable import Variable


def test_process_pedestal_profiles_preserve_average_and_edge_values() -> None:
    rho = np.linspace(0.0, 1.0, 1001)
    temperature = RELATIONS.get("PROCESS pedestal electron temperature profile").evaluate(
        {
            "T_e_avg": 10.0,
            "temp_plasma_pedestal_kev": 4.0,
            "T_sep": 0.1,
            "radius_plasma_pedestal_temp_norm": 0.9,
            "alphat": 0.5,
            "tbeta": 2.0,
            "rho_minor": rho,
        }
    )
    density = RELATIONS.get("PROCESS pedestal electron density profile").evaluate(
        {
            "n_e_avg": 1.0e20,
            "nd_plasma_pedestal_electron": 7.0e19,
            "n_sep": 2.0e19,
            "radius_plasma_pedestal_density_norm": 0.9,
            "alphan": 0.25,
            "rho_minor": rho,
        }
    )

    assert temperature[-1] == pytest.approx(0.1)
    assert density[-1] == pytest.approx(2.0e19)
    assert volume_average(temperature, rho) == pytest.approx(10.0, rel=3e-5)
    assert volume_average(density, rho) == pytest.approx(1.0e20, rel=1e-4)


def _profile_reactor(mode: str) -> Reactor:
    values = {
        "T_e_avg": 10.0,
        "T_i_avg": 11.0,
        "temp_plasma_pedestal_kev": 4.0,
        "T_sep": 0.1,
        "radius_plasma_pedestal_temp_norm": 0.9,
        "n_e_avg": 1.0e20,
        "n_fuel_avg": 8.0e19,
        "nd_plasma_pedestal_electron": 7.0e19,
        "n_sep": 2.0e19,
        "radius_plasma_pedestal_density_norm": 0.9,
        "temperature_peaking": 2.0,
        "ion_temperature_peaking": 2.0,
        "density_peaking": 1.4,
        "ion_density_peaking": 1.4,
    }
    variables = {name: Variable(name, value) for name, value in values.items()}
    return Reactor(name=f"{mode} profile selection", tags=("tokamak", mode), variables=variables)


def test_mode_profile_defaults_encode_h_l_i_boundary_conditions() -> None:
    h_shapes = set(_profile_reactor("h_mode")._supplied_shape_includes())
    i_shapes = set(_profile_reactor("i_mode")._supplied_shape_includes())
    l_shapes = set(_profile_reactor("l_mode")._supplied_shape_includes())

    assert "PROCESS pedestal electron temperature profile" in h_shapes
    assert "PROCESS pedestal electron density profile" in h_shapes
    assert "PROCESS pedestal electron temperature profile" in i_shapes
    assert "PROCESS pedestal electron density profile" not in i_shapes
    assert "Parabolic electron density profile" in i_shapes
    assert "PROCESS pedestal electron temperature profile" not in l_shapes
    assert "PROCESS pedestal electron density profile" not in l_shapes
    assert "Parabolic electron temperature profile" in l_shapes
    assert "Parabolic electron density profile" in l_shapes


def test_fuse_imas_hmode_profile_is_selectable_and_has_exact_endpoints() -> None:
    rho_tor = np.linspace(0.0, 1.0, 101)
    relation = RELATIONS.get("FUSE IMAS H-mode electron temperature profile")
    profile = relation.evaluate(
        {
            "T_sep": 0.08,
            "temp_plasma_pedestal_kev": 3.0,
            "T0": 18.0,
            "alphat": 1.2,
            "pedestal_width": 0.08,
            "rho_tor": rho_tor,
        }
    )
    assert profile[0] == pytest.approx(18.0)
    assert profile[-1] == pytest.approx(0.08)
    assert {"h_mode", "i_mode"} <= set(relation.tags)
    assert "i_mode" not in RELATIONS.get("FUSE IMAS H-mode electron density profile").tags


def test_fuse_imas_hmode_profile_uses_explicit_toroidal_flux_coordinate() -> None:
    rho = np.linspace(0.0, 1.0, 101)
    relation = RELATIONS.get("FUSE IMAS H-mode electron temperature profile")
    values = {
        "T_sep": 0.08,
        "temp_plasma_pedestal_kev": 3.0,
        "T0": 18.0,
        "alphat": 1.2,
        "pedestal_width": 0.08,
    }

    identity = relation.evaluate({**values, "rho_tor": rho})
    mapped = relation.evaluate({**values, "rho_tor": rho**1.4})

    assert "rho_tor" in relation.input_names
    assert "rho" not in relation.input_names
    assert identity.shape == mapped.shape
    assert not np.allclose(identity[1:-1], mapped[1:-1])
    assert mapped[0] == pytest.approx(18.0)
    assert mapped[-1] == pytest.approx(0.08)


def test_tokamak_toroidal_flux_coordinate_is_identity_fallback() -> None:
    rho = np.linspace(0.0, 1.0, 31)
    relation = RELATIONS.get("Tokamak normalized toroidal-flux coordinate")

    mapped = relation.evaluate({"rho": rho})

    assert np.array_equal(mapped, rho)
    assert VARIABLES.get("rho_tor").default_relation[0] == relation.name


def test_fuse_imas_hmode_accessibility_is_a_checked_only_guard() -> None:
    guard = RELATIONS.get("H-mode accessibility (diverted, non-negative triangularity)")
    assert not guard.enforce
    assert guard.evaluate({"has_x_point": 1.0, "delta": 0.2}) == pytest.approx(0.0)
    assert guard.evaluate({"has_x_point": 0.0, "delta": 0.2}) > 0.0
    assert guard.evaluate({"has_x_point": 1.0, "delta": -0.2}) > 0.0


def test_fuse_imas_lh_threshold_applies_drift_wall_and_rollover_corrections() -> None:
    relation = RELATIONS.get("L-H threshold power (FUSE IMAS corrected)")
    base = {
        "n_la": 1.0e20,
        "I_p": 8.0e6,
        "B0": 5.0,
        "R": 6.0,
        "a": 2.0,
        "A_p": 700.0,
        "afuel": 2.5,
        "B0_sign": -1.0,
        "x_point_z": -3.0,
        "is_metallic_wall": 1.0,
    }
    favorable_metal = relation.evaluate(base)
    unfavorable = relation.evaluate({**base, "x_point_z": 3.0})
    carbon = relation.evaluate({**base, "is_metallic_wall": 0.0})
    density_clamped_1 = relation.evaluate({**base, "n_la": 1.0e10})
    density_clamped_2 = relation.evaluate({**base, "n_la": 1.0e15})

    assert unfavorable == pytest.approx(2.0 * favorable_metal)
    assert carbon == pytest.approx(favorable_metal / 0.8)
    assert density_clamped_1 == pytest.approx(density_clamped_2)


def test_process_fast_alpha_beta_and_fuse_total_pressure_aggregation() -> None:
    values = {
        "B_total": 5.2,
        "n_e_avg": 1.0e20,
        "n_fuel_avg": 8.0e19,
        "n_i_avg": 9.0e19,
        "temp_plasma_electron_density_weighted": 15.0,
        "temp_plasma_ion_density_weighted": 15.0,
        "P_fus_DT_alpha": 100.0e6,
        "P_alpha_beam": 10.0e6,
        "f_D": 0.5,
    }
    beta_fast = RELATIONS.get("Fast-alpha beta Ward (PROCESS)").evaluate(values)
    beta_ipdg89 = RELATIONS.get("Fast-alpha beta IPDG89 (PROCESS)").evaluate(values)
    beta_thermal = (
        2.0
        * MU0
        * KEV_TO_J
        * (values["n_e_avg"] * 15.0 + values["n_i_avg"] * 15.0)
        / values["B_total"] ** 2
    )
    ward_fraction = min(0.30, 0.26 * 0.8**2 * np.sqrt(30.0 / 20.0 - 0.65))
    assert beta_fast == pytest.approx(beta_thermal * ward_fraction * 1.1)
    assert beta_ipdg89 == pytest.approx(beta_thermal * 0.29 * 0.8**2 * (30.0 / 20.0 - 0.37) * 1.1)
    assert VARIABLES.get("beta_fast_alpha").default_relation == ("Fast-alpha beta Ward (PROCESS)",)
    assert "i_beta_fast_alpha" not in VARIABLES

    p_fast = RELATIONS.get("Fast-alpha pressure from beta").evaluate(
        {"beta_fast_alpha": beta_fast, "B_total": values["B_total"]}
    )
    total = RELATIONS.get("Total pressure including fast ions (FUSE)").evaluate(
        {"p_th": 2.0e6, "p_fast_alpha": p_fast, "p_fast_beam": 3.0e4}
    )
    assert total == pytest.approx(2.03e6 + p_fast)


def test_wall_surface_defaults_to_plasma_surface_and_drives_wall_loading() -> None:
    fallback = RELATIONS.get("Wall surface from plasma surface")
    loading = RELATIONS.get("Neutron wall loading")
    assert VARIABLES.get("S_wall").default_relation == ("Wall surface from plasma surface",)
    assert fallback.evaluate({"A_p": 600.0}) == pytest.approx(600.0)
    assert loading.evaluate({"P_neutron": 1.2e9, "S_wall": 800.0}) == pytest.approx(1.5e6)


def test_imported_relations_carry_source_notes() -> None:
    names_and_sources = {
        "PROCESS pedestal electron temperature profile": "PROCESS",
        "FUSE IMAS H-mode electron temperature profile": "FUSE/IMAS",
        "Fast-alpha beta Ward (PROCESS)": "PROCESS",
        "Total pressure including fast ions (FUSE)": "FUSE",
    }
    for name, source in names_and_sources.items():
        assert source in (RELATIONS.get(name).func.__doc__ or "")
