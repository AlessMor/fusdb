import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relation_util import get_filtered_relations
from fusdb.registry.reactor_defaults import apply_reactor_defaults
from fusdb.relations.reactivities.reactivity_functions import (
    sigmav_DD_total_NRL,
    sigmav_DHe3_NRL,
    sigmav_DT_NRL,
    sigmav_DD_ENDFB_VIII0,
    sigmav_DHe3_ENDFB_VIII0,
    sigmav_DT_ENDFB_VIII0,
    sigmav_He3He3_CF88,
    sigmav_He3He3_ENDFB_VIII0,
    sigmav_THe3_CF88,
    sigmav_THe3_D_CF88,
    sigmav_THe3_D_ENDFB_VIII0,
    sigmav_THe3_He5p,
    sigmav_THe3_np_CF88,
    sigmav_THe3_np_ENDFB_VIII0,
    sigmav_THe3_ENDFB_VIII0,
    sigmav_THe3_NRL,
    sigmav_TT_CF88,
    sigmav_TT_ENDFB_VIII0,
    sigmav_TT_NRL,
)
from fusdb.relations.reactivities.reactivity_profile import (
    sigmav_ddn_profile,
    sigmav_ddn_profile_endfb_viii0,
    sigmav_ddp_profile,
    sigmav_ddp_profile_endfb_viii0,
    sigmav_dhe3_profile,
    sigmav_dhe3_profile_endfb_viii0,
    sigmav_dhe3_profile_nrl,
    sigmav_dt_profile,
    sigmav_dt_profile_endfb_viii0,
    sigmav_dt_profile_nrl,
    sigmav_he3he3_profile,
    sigmav_he3he3_profile_cf88,
    sigmav_he3he3_profile_endfb_viii0,
    sigmav_the3_profile,
    sigmav_the3_profile_cf88,
    sigmav_the3_profile_endfb_viii0,
    sigmav_the3_profile_nrl,
    sigmav_tt_profile,
    sigmav_tt_profile_cf88,
    sigmav_tt_profile_endfb_viii0,
    sigmav_tt_profile_nrl,
)
from fusdb.variable_util import make_variable


def test_cf88_scalar_reactivities_are_positive():
    """Expected: implemented CF88 scalar reactivities are finite and non-negative."""
    temp_keV = 10.0

    tt = float(sigmav_TT_CF88(temp_keV))
    he3he3 = float(sigmav_He3He3_CF88(temp_keV))
    the3_np = float(sigmav_THe3_np_CF88(temp_keV))
    the3_d = float(sigmav_THe3_D_CF88(temp_keV))
    the3_he5p = float(sigmav_THe3_He5p(temp_keV))

    assert math.isfinite(tt) and tt > 0.0
    assert math.isfinite(he3he3) and he3he3 > 0.0
    assert math.isfinite(the3_np) and the3_np > 0.0
    assert math.isfinite(the3_d) and the3_d > 0.0
    assert the3_he5p == 0.0


def test_cf88_array_reactivities_preserve_shape_and_total_the3_sum():
    """Expected: CF88 array helpers preserve shape and the total T-He3 helper sums branches."""
    temp_keV = np.asarray([5.0, 10.0, 20.0], dtype=float)

    tt = np.asarray(sigmav_TT_CF88(temp_keV), dtype=float)
    he3he3 = np.asarray(sigmav_He3He3_CF88(temp_keV), dtype=float)
    the3_np, the3_d, the3_he5p = sigmav_THe3_CF88(temp_keV)
    the3_total = (
        np.asarray(the3_np, dtype=float)
        + np.asarray(the3_d, dtype=float)
        + np.asarray(the3_he5p, dtype=float)
    )

    assert tt.shape == temp_keV.shape
    assert he3he3.shape == temp_keV.shape
    assert the3_total.shape == temp_keV.shape
    assert np.all(tt > 0.0)
    assert np.all(he3he3 > 0.0)
    assert np.all(the3_total > 0.0)


def test_cf88_profile_relations_expose_symbolic_models():
    """Expected: profile relations backed by CF88 helpers still register sympy expressions."""
    assert sigmav_tt_profile_cf88.sympy_expression is not None
    assert sigmav_he3he3_profile_cf88.sympy_expression is not None
    assert sigmav_the3_profile_cf88.sympy_expression is not None


def test_endfb_scalar_reactivities_are_positive():
    """Expected: ENDF/B-VIII.0 tabulated scalar reactivities are finite and non-negative."""
    temp_keV = 10.0

    dt = float(sigmav_DT_ENDFB_VIII0(temp_keV))
    dd_total, ddn, ddp = sigmav_DD_ENDFB_VIII0(temp_keV)
    dhe3 = float(sigmav_DHe3_ENDFB_VIII0(temp_keV))
    tt = float(sigmav_TT_ENDFB_VIII0(temp_keV))
    he3he3 = float(sigmav_He3He3_ENDFB_VIII0(temp_keV))
    the3_np = float(sigmav_THe3_np_ENDFB_VIII0(temp_keV))
    the3_d = float(sigmav_THe3_D_ENDFB_VIII0(temp_keV))
    the3_total = float(sigmav_THe3_ENDFB_VIII0(temp_keV))

    assert math.isfinite(dt) and dt > 0.0
    assert math.isfinite(float(dd_total)) and float(dd_total) > 0.0
    assert math.isfinite(float(ddn)) and float(ddn) > 0.0
    assert math.isfinite(float(ddp)) and float(ddp) > 0.0
    assert math.isclose(float(dd_total), float(ddn) + float(ddp), rel_tol=1e-12)
    assert math.isfinite(dhe3) and dhe3 > 0.0
    assert math.isfinite(tt) and tt > 0.0
    assert math.isfinite(he3he3) and he3he3 > 0.0
    assert math.isfinite(the3_np) and the3_np > 0.0
    assert math.isfinite(the3_d) and the3_d > 0.0
    assert math.isclose(the3_total, the3_np + the3_d, rel_tol=1e-12)


def test_endfb_array_reactivities_preserve_shape_and_branch_sums():
    """Expected: ENDF/B-VIII.0 array helpers preserve shape and branch sums."""
    temp_keV = np.asarray([5.0, 10.0, 20.0], dtype=float)

    dt = np.asarray(sigmav_DT_ENDFB_VIII0(temp_keV), dtype=float)
    dd_total, ddn, ddp = sigmav_DD_ENDFB_VIII0(temp_keV)
    dhe3 = np.asarray(sigmav_DHe3_ENDFB_VIII0(temp_keV), dtype=float)
    tt = np.asarray(sigmav_TT_ENDFB_VIII0(temp_keV), dtype=float)
    he3he3 = np.asarray(sigmav_He3He3_ENDFB_VIII0(temp_keV), dtype=float)
    the3_total = np.asarray(sigmav_THe3_ENDFB_VIII0(temp_keV), dtype=float)
    the3_np = np.asarray(sigmav_THe3_np_ENDFB_VIII0(temp_keV), dtype=float)
    the3_d = np.asarray(sigmav_THe3_D_ENDFB_VIII0(temp_keV), dtype=float)

    assert dt.shape == temp_keV.shape
    assert np.asarray(dd_total, dtype=float).shape == temp_keV.shape
    assert np.asarray(ddn, dtype=float).shape == temp_keV.shape
    assert np.asarray(ddp, dtype=float).shape == temp_keV.shape
    assert dhe3.shape == temp_keV.shape
    assert tt.shape == temp_keV.shape
    assert he3he3.shape == temp_keV.shape
    assert the3_total.shape == temp_keV.shape
    assert np.all(dt > 0.0)
    assert np.all(np.asarray(dd_total, dtype=float) > 0.0)
    assert np.all(dhe3 > 0.0)
    assert np.all(tt > 0.0)
    assert np.all(he3he3 > 0.0)
    assert np.allclose(np.asarray(dd_total, dtype=float), np.asarray(ddn, dtype=float) + np.asarray(ddp, dtype=float))
    assert np.allclose(the3_total, the3_np + the3_d)


def test_endfb_profile_relations_expose_symbolic_models():
    """Expected: ENDF/B-VIII.0 profile relations also register symbolic expressions."""
    assert sigmav_dt_profile_endfb_viii0.sympy_expression is not None
    assert sigmav_ddn_profile_endfb_viii0.sympy_expression is not None
    assert sigmav_ddp_profile_endfb_viii0.sympy_expression is not None
    assert sigmav_dhe3_profile_endfb_viii0.sympy_expression is not None
    assert sigmav_tt_profile_endfb_viii0.sympy_expression is not None
    assert sigmav_he3he3_profile_endfb_viii0.sympy_expression is not None
    assert sigmav_the3_profile_endfb_viii0.sympy_expression is not None


def test_nrl_profile_relations_expose_symbolic_models_without_numeric_rows():
    """Expected: NRL tabulated profile relations register symbolic expressions before tables are populated."""
    assert sigmav_dt_profile_nrl.sympy_expression is not None
    assert sigmav_dhe3_profile_nrl.sympy_expression is not None
    assert sigmav_tt_profile_nrl.sympy_expression is not None
    assert sigmav_the3_profile_nrl.sympy_expression is not None


def test_nrl_reactivities_default_to_pchip_interpolation():
    """Expected: NRL helper defaults match explicitly requested PCHIP interpolation."""
    temp_keV = np.asarray([3.0, 30.0, 300.0], dtype=float)

    assert np.allclose(
        np.asarray(sigmav_DT_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_DT_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )
    assert np.allclose(
        np.asarray(sigmav_DD_total_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_DD_total_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )
    assert np.allclose(
        np.asarray(sigmav_DHe3_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_DHe3_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )
    assert np.allclose(
        np.asarray(sigmav_TT_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_TT_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )
    assert np.allclose(
        np.asarray(sigmav_THe3_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_THe3_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )


def test_nrl_reactivities_accept_selectable_interpolation_kind_and_reject_invalid_values():
    """Expected: NRL helpers accept shape-preserving and spline interpolation kinds and reject unsupported ones."""
    temp_keV = np.asarray([3.0, 30.0, 300.0], dtype=float)

    linear = np.asarray(sigmav_DT_NRL(temp_keV, interpolation_kind="linear"), dtype=float)
    pchip = np.asarray(sigmav_DT_NRL(temp_keV, interpolation_kind="pchip"), dtype=float)
    quadratic = np.asarray(sigmav_DT_NRL(temp_keV, interpolation_kind="quadratic"), dtype=float)

    assert linear.shape == temp_keV.shape
    assert pchip.shape == temp_keV.shape
    assert quadratic.shape == temp_keV.shape
    assert np.all(linear > 0.0)
    assert np.all(pchip > 0.0)
    assert np.all(quadratic > 0.0)
    assert not np.allclose(linear, pchip, rtol=1e-6, atol=0.0)
    assert not np.allclose(pchip, quadratic, rtol=1e-6, atol=0.0)

    try:
        sigmav_DT_NRL(temp_keV, interpolation_kind="bad-kind")
    except ValueError as exc:
        assert "interpolation_kind" in str(exc)
    else:
        raise AssertionError("Expected unsupported interpolation_kind to raise ValueError")


def test_reactivity_profile_relations_have_method_specific_names():
    """Expected: each reactivity-profile relation name encodes the selectable method."""
    assert sigmav_dt_profile.name == "DT reactivity profile BoschHale"
    assert sigmav_dt_profile_endfb_viii0.name == "DT reactivity profile ENDFB-VIII0"
    assert sigmav_dt_profile_nrl.name == "DT reactivity profile NRL"
    assert sigmav_ddn_profile.name == "DDn reactivity profile BoschHale"
    assert sigmav_ddn_profile_endfb_viii0.name == "DDn reactivity profile ENDFB-VIII0"
    assert sigmav_ddp_profile.name == "DDp reactivity profile BoschHale"
    assert sigmav_ddp_profile_endfb_viii0.name == "DDp reactivity profile ENDFB-VIII0"
    assert sigmav_dhe3_profile.name == "DHe3 reactivity profile BoschHale"
    assert sigmav_dhe3_profile_endfb_viii0.name == "DHe3 reactivity profile ENDFB-VIII0"
    assert sigmav_dhe3_profile_nrl.name == "DHe3 reactivity profile NRL"
    assert sigmav_tt_profile.name == "TT reactivity profile"
    assert sigmav_tt_profile_cf88.name == "TT reactivity profile CF88"
    assert sigmav_tt_profile_endfb_viii0.name == "TT reactivity profile ENDFB-VIII0"
    assert sigmav_tt_profile_nrl.name == "TT reactivity profile NRL"
    assert sigmav_he3he3_profile.name == "He3He3 reactivity profile"
    assert sigmav_he3he3_profile_cf88.name == "He3He3 reactivity profile CF88"
    assert sigmav_he3he3_profile_endfb_viii0.name == "He3He3 reactivity profile ENDFB-VIII0"
    assert sigmav_the3_profile.name == "THe3 reactivity profile"
    assert sigmav_the3_profile_cf88.name == "THe3 reactivity profile CF88"
    assert sigmav_the3_profile_endfb_viii0.name == "THe3 reactivity profile ENDFB-VIII0"
    assert sigmav_the3_profile_nrl.name == "THe3 reactivity profile NRL"


def test_reactor_defaults_assign_reactivity_profile_methods_without_overwriting_user_choice():
    """Expected: reactor defaults create reactivity-profile variables with default methods but preserve explicit overrides."""
    variables = {
        "sigmav_DT_profile": make_variable(
            name="sigmav_DT_profile",
            ndim=1,
            method="custom dt method",
        ),
    }

    apply_reactor_defaults(variables)

    assert variables["sigmav_DT_profile"].method == "custom dt method"
    assert variables["sigmav_DDn_profile"].method == "DDn reactivity profile BoschHale"
    assert variables["sigmav_DDp_profile"].method == "DDp reactivity profile BoschHale"
    assert variables["sigmav_DHe3_profile"].method == "DHe3 reactivity profile BoschHale"
    assert variables["sigmav_TT_profile"].method == "TT reactivity profile"
    assert variables["sigmav_He3He3_profile"].method == "He3He3 reactivity profile"
    assert variables["sigmav_THe3_profile"].method == "THe3 reactivity profile"


def test_method_override_filters_reactivity_relations_by_name():
    """Expected: method selection keeps the requested reactivity relation and rejects others with the same output."""
    rels = get_filtered_relations(
        ("fusion_power",),
        ("T_i", "sigmav_TT_profile"),
        ("TT reactivity profile ENDFB-VIII0",),
    )
    names = {
        rel.name
        for rel in rels
        if rel.preferred_target == "sigmav_TT_profile"
    }
    assert "TT reactivity profile ENDFB-VIII0" in names
    assert "TT reactivity profile" not in names
    assert "TT reactivity profile CF88" not in names
