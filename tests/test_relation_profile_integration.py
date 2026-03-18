import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relations.reactivities.reactivity_functions import sigmav_DT_BoschHale
from fusdb.relations.reactivities.reactivity_functions import sigmav_THe3_D_CF88
from fusdb.relations.reactivities.reactivity_functions import sigmav_THe3_np_CF88
from fusdb.relations.power_balance.fusion_power.reaction_rate import reaction_rate_dt
from fusdb.relations.power_balance.fusion_power.reaction_rate import reaction_rate_the3
from fusdb.relations.power_balance.fusion_power.reaction_rate import reaction_rate_the3_d
from fusdb.relations.power_balance.fusion_power.reaction_rate import reaction_rate_the3_np
from fusdb.relations.plasma_pressure.plasma_pressure import thermal_pressure
from fusdb.relations.plasma_composition import plasma_composition as composition_relations
from fusdb.relationsystem_class import RelationSystem
from fusdb.relation_util import _RELATION_REGISTRY, relation
from fusdb.utils import compare_plasma_volume_with_integrated_dv, integrate_profile_over_volume
from fusdb.variable_util import make_variable


def _make_var(name: str, value: object, *, ndim: int) -> object:
    var = make_variable(name=name, ndim=ndim)
    var.add_value(value, as_input=True)
    var.input_source = "explicit"
    return var


def test_reaction_rate_total_from_profile_pipeline_matches_expected():
    """Expected: profile-based reaction-rate pipeline integrates to the same total as direct manual integration."""
    n_i_profile = np.full(51, 1.0e20, dtype=float)
    t_i_profile = np.full(51, 10.0, dtype=float)
    f_d = 0.5
    f_t = 0.5
    v_p = 100.0

    variables = [
        _make_var("f_D", f_d, ndim=0),
        _make_var("f_T", f_t, ndim=0),
        _make_var("n_i", n_i_profile, ndim=1),
        _make_var("T_i", t_i_profile, ndim=1),
        _make_var("V_p", v_p, ndim=0),
    ]
    system = RelationSystem([sigmav_DT_BoschHale, reaction_rate_dt], variables, mode="overwrite")
    system.solve()

    sigmav_profile = system.variables_dict["sigmav_DT"].current_value
    rr = system.variables_dict["Rr_DT"].current_value

    expected_integrand = f_d * f_t * (n_i_profile ** 2) * np.asarray(sigmav_DT_BoschHale(t_i_profile), dtype=float)
    expected = integrate_profile_over_volume(expected_integrand, v_p)

    assert sigmav_profile is not None
    assert np.asarray(sigmav_profile, dtype=float).shape == t_i_profile.shape
    assert rr is not None
    assert expected is not None
    assert math.isclose(float(rr), float(expected), rel_tol=1e-12, abs_tol=0.0)


def test_the3_reaction_rate_pipeline_sums_branch_rates():
    """Expected: T-He3 total reaction rate is assembled from the two branch-rate relations."""
    n_i_profile = np.full(31, 2.0e19, dtype=float)
    t_i_profile = np.full(31, 20.0, dtype=float)
    f_t = 0.4
    f_he3 = 0.1
    v_p = 50.0

    variables = [
        _make_var("f_T", f_t, ndim=0),
        _make_var("f_He3", f_he3, ndim=0),
        _make_var("n_i", n_i_profile, ndim=1),
        _make_var("T_i", t_i_profile, ndim=1),
        _make_var("V_p", v_p, ndim=0),
    ]
    system = RelationSystem(
        [sigmav_THe3_D_CF88, sigmav_THe3_np_CF88, reaction_rate_the3_d, reaction_rate_the3_np, reaction_rate_the3],
        variables,
        mode="overwrite",
    )
    system.solve()

    rr_d = system.variables_dict["Rr_THe3_D"].current_value
    rr_np = system.variables_dict["Rr_THe3_np"].current_value
    rr_total = system.variables_dict["Rr_THe3"].current_value

    assert rr_d is not None
    assert rr_np is not None
    assert rr_total is not None
    assert math.isclose(float(rr_total), float(rr_d) + float(rr_np), rel_tol=1e-12, abs_tol=0.0)


def test_scalar_output_relation_returning_profile_is_not_auto_integrated():
    """Expected: a scalar-output relation returning a profile is left unsolved (no implicit auto-integration)."""
    @relation(name="profile_without_explicit_integral", output="Rr_DT", tags=("test",))
    def profile_without_explicit_integral(n_i: float) -> float:
        return n_i

    try:
        variables = [_make_var("n_i", np.full(11, 1.0, dtype=float), ndim=1)]
        system = RelationSystem([profile_without_explicit_integral], variables, mode="overwrite", max_passes=0)
        system.solve()

        out = system._graph["vars"].get("Rr_DT")
        assert out is None or out.current_value is None
    finally:
        if _RELATION_REGISTRY and _RELATION_REGISTRY[-1] is profile_without_explicit_integral:
            _RELATION_REGISTRY.pop()


def test_numeric_inverse_fallback_solves_single_scalar_unknown():
    """Expected: numeric inverse fallback solves a single scalar unknown when the inverse problem is bracketed."""
    @relation(name="integrated_linear_rate", output="Rr_DT", tags=("test",))
    def integrated_linear_rate(k: float, n_i: float, V_p: float) -> float:
        total = integrate_profile_over_volume(n_i, V_p)
        if total is None:
            raise ValueError("Integration failed")
        return k * total

    try:
        profile = np.full(31, 2.0, dtype=float)
        v_p = 10.0
        target = 60.0

        k_var = make_variable(name="k", ndim=0)
        variables = [
            _make_var("n_i", profile, ndim=1),
            _make_var("V_p", v_p, ndim=0),
            _make_var("Rr_DT", target, ndim=0),
            k_var,
        ]
        system = RelationSystem([integrated_linear_rate], variables, mode="overwrite", max_passes=0)
        system.solve()

        solved_k = system.variables_dict["k"].current_value
        assert solved_k is not None
        assert math.isclose(float(solved_k), 3.0, rel_tol=1e-8, abs_tol=1e-8)
    finally:
        if _RELATION_REGISTRY and _RELATION_REGISTRY[-1] is integrated_linear_rate:
            _RELATION_REGISTRY.pop()


def test_numeric_inverse_fallback_leaves_unsolved_when_unbracketed():
    """Expected: numeric inverse fallback leaves the unknown unsolved when the requested target is unphysical/unbracketed."""
    @relation(name="bounded_integrated_rate", output="Rr_DT", tags=("test",))
    def bounded_integrated_rate(f_D: float, n_i: float, V_p: float) -> float:
        total = integrate_profile_over_volume(f_D * n_i, V_p)
        if total is None:
            raise ValueError("Integration failed")
        return total

    try:
        f_d_var = make_variable(name="f_D", ndim=0)
        variables = [
            _make_var("n_i", np.full(21, 1.0, dtype=float), ndim=1),
            _make_var("V_p", 5.0, ndim=0),
            _make_var("Rr_DT", -1.0, ndim=0),  # impossible with f_D in [0, 1]
            f_d_var,
        ]
        system = RelationSystem([bounded_integrated_rate], variables, mode="overwrite", max_passes=0)
        system.solve()

        out = system.variables_dict.get("f_D")
        assert out is not None
        assert out.current_value is None
    finally:
        if _RELATION_REGISTRY and _RELATION_REGISTRY[-1] is bounded_integrated_rate:
            _RELATION_REGISTRY.pop()


def test_profile_input_inverse_does_not_use_profile_mean_fallback():
    """Expected: inverse solving does not silently reduce profile inputs to means for scalar unknown recovery."""
    @relation(name="profile_affine_output", output="Rr_DT", tags=("test",))
    def profile_affine_output(k: float, n_i: float) -> float:
        return k * n_i

    try:
        k_var = make_variable(name="k", ndim=0)
        variables = [
            _make_var("n_i", np.asarray([1.0, 3.0, 5.0], dtype=float), ndim=1),
            _make_var("Rr_DT", 10.0, ndim=0),
            k_var,
        ]
        system = RelationSystem([profile_affine_output], variables, mode="overwrite", max_passes=0)
        system.solve()

        # Strict behavior: no hidden mean(profile) scalarization for inverse solving.
        out = system.variables_dict.get("k")
        assert out is not None
        assert out.current_value is None
    finally:
        if _RELATION_REGISTRY and _RELATION_REGISTRY[-1] is profile_affine_output:
            _RELATION_REGISTRY.pop()


def test_compare_volume_with_integrated_dv_warns_when_geometry_mismatch():
    """Expected: geometry/volume mismatch triggers a warning and reports non-matching integrated/reference volumes."""
    with pytest.warns(UserWarning):
        ok, v_int, v_ref = compare_plasma_volume_with_integrated_dv(
            V_p=100.0,
            R=3.0,
            a=1.0,
            kappa=2.0,
            rel_tol=0.01,
            warn=True,
        )
    assert ok is False
    assert v_int is not None and v_ref is not None


def test_reaction_rate_symbolic_model_is_available():
    """Expected: reaction-rate relation exposes a symbolic expression model."""
    assert reaction_rate_dt.sympy_expression is not None


def test_thermal_pressure_symbolic_model_is_available():
    """Expected: thermal-pressure relation exposes a symbolic expression model."""
    assert thermal_pressure.sympy_expression is not None


def test_fraction_equilibrium_symbolic_models_are_available():
    """Expected: all fraction-equilibrium relations built on FRACTIONS inputs expose symbolic expressions."""
    fractions = tuple(composition_relations.FRACTIONS)
    rels = [
        rel
        for rel in composition_relations._relations
        if tuple(rel.required_inputs()) == fractions
    ]
    assert rels
    assert all(rel.sympy_expression is not None for rel in rels)
