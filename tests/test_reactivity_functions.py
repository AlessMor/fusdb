import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relation_util import get_filtered_relations
from fusdb.registry import allowed_variable_ndim
from fusdb.registry.reactor_defaults import apply_reactor_defaults
from fusdb.relations.reactivities.reactivity_functions import (
    sigmav_DD_BoschHale,
    sigmav_DD_ENDFB_VIII1,
    sigmav_DD_ENDFB_VIII0,
    sigmav_DD_Hively,
    sigmav_DD_NRL,
    sigmav_DDn_BoschHale,
    sigmav_DDn_ENDFB_VIII1,
    sigmav_DDn_Hively,
    sigmav_DDp_BoschHale,
    sigmav_DDp_ENDFB_VIII1,
    sigmav_DDp_Hively,
    sigmav_DHe3_NRL,
    sigmav_DHe3_BoschHale,
    sigmav_DHe3_ENDFB_VIII1,
    sigmav_DDn_ENDFB_VIII0,
    sigmav_DDp_ENDFB_VIII0,
    sigmav_DT_Hively,
    sigmav_DT_NRL,
    sigmav_DT_BoschHale,
    sigmav_DT_ENDFB_VIII1,
    sigmav_DHe3_ENDFB_VIII0,
    sigmav_DT_ENDFB_VIII0,
    sigmav_He3He3_CF88,
    sigmav_He3He3_ENDFB_VIII1,
    sigmav_He3He3_ENDFB_VIII0,
    sigmav_THe3_D_CF88,
    sigmav_THe3_D_ENDFB_VIII0,
    sigmav_THe3_D_ENDFB_VIII1,
    sigmav_THe3_CF88,
    sigmav_THe3_ENDFB_VIII1,
    sigmav_THe3_ENDFB_VIII0,
    sigmav_THe3_NRL,
    sigmav_THe3_np_CF88,
    sigmav_THe3_np_ENDFB_VIII0,
    sigmav_THe3_np_ENDFB_VIII1,
    sigmav_TT_CF88,
    sigmav_TT_ENDFB_VIII1,
    sigmav_TT_ENDFB_VIII0,
    sigmav_TT_NRL,
)
from fusdb.variable_util import make_variable


def test_cf88_scalar_reactivities_are_positive():
    """Expected: implemented CF88 scalar reactivities are finite and non-negative."""
    temp_keV = 10.0

    tt = float(sigmav_TT_CF88(temp_keV))
    he3he3 = float(sigmav_He3He3_CF88(temp_keV))
    the3_np = float(sigmav_THe3_np_CF88(temp_keV))
    the3_d = float(sigmav_THe3_D_CF88(temp_keV))

    assert math.isfinite(tt) and tt > 0.0
    assert math.isfinite(he3he3) and he3he3 > 0.0
    assert math.isfinite(the3_np) and the3_np > 0.0
    assert math.isfinite(the3_d) and the3_d > 0.0


def test_cf88_array_reactivities_preserve_shape_and_total_the3_sum():
    """Expected: CF88 array helpers preserve shape and the total T-He3 helper sums branches."""
    temp_keV = np.asarray([5.0, 10.0, 20.0], dtype=float)

    tt = np.asarray(sigmav_TT_CF88(temp_keV), dtype=float)
    he3he3 = np.asarray(sigmav_He3He3_CF88(temp_keV), dtype=float)
    the3_total = np.asarray(sigmav_THe3_CF88(temp_keV), dtype=float)
    the3_branch_sum = (
        np.asarray(sigmav_THe3_np_CF88(temp_keV), dtype=float)
        + np.asarray(sigmav_THe3_D_CF88(temp_keV), dtype=float)
    )

    assert tt.shape == temp_keV.shape
    assert he3he3.shape == temp_keV.shape
    assert the3_total.shape == temp_keV.shape
    assert np.all(tt > 0.0)
    assert np.all(he3he3 > 0.0)
    assert np.all(the3_total > 0.0)
    assert np.allclose(the3_total, the3_branch_sum)


def test_cf88_reactivity_relations_expose_symbolic_models():
    """Expected: CF88-backed reactivity relations still register sympy expressions."""
    assert sigmav_TT_CF88.sympy_expression is not None
    assert sigmav_He3He3_CF88.sympy_expression is not None
    assert sigmav_THe3_CF88.sympy_expression is not None


def test_hively_scalar_reactivities_are_positive_and_dd_branches_sum():
    """Expected: Hively scalar reactivities are finite, positive, and DD branches add up."""
    temp_keV = 10.0

    dt = float(sigmav_DT_Hively(temp_keV))
    dd_total = float(sigmav_DD_Hively(temp_keV))
    ddn = float(sigmav_DDn_Hively(temp_keV))
    ddp = float(sigmav_DDp_Hively(temp_keV))

    assert math.isfinite(dt) and dt > 0.0
    assert math.isfinite(dd_total) and dd_total > 0.0
    assert math.isfinite(ddn) and ddn > 0.0
    assert math.isfinite(ddp) and ddp > 0.0
    assert math.isclose(dd_total, ddn + ddp, rel_tol=1e-12)


def test_hively_array_reactivities_preserve_shape_and_dd_branch_sums():
    """Expected: Hively array helpers preserve shape and DD branch sums."""
    temp_keV = np.asarray([5.0, 10.0, 20.0], dtype=float)

    dt = np.asarray(sigmav_DT_Hively(temp_keV), dtype=float)
    dd_total = np.asarray(sigmav_DD_Hively(temp_keV), dtype=float)
    ddn = np.asarray(sigmav_DDn_Hively(temp_keV), dtype=float)
    ddp = np.asarray(sigmav_DDp_Hively(temp_keV), dtype=float)

    assert dt.shape == temp_keV.shape
    assert dd_total.shape == temp_keV.shape
    assert ddn.shape == temp_keV.shape
    assert ddp.shape == temp_keV.shape
    assert np.all(dt > 0.0)
    assert np.all(dd_total > 0.0)
    assert np.allclose(dd_total, ddn + ddp)


def test_hively_reactivity_relations_expose_symbolic_models():
    """Expected: Hively reactivity relations also register symbolic expressions."""
    assert sigmav_DT_Hively.sympy_expression is not None
    assert sigmav_DDn_Hively.sympy_expression is not None
    assert sigmav_DDp_Hively.sympy_expression is not None


def test_endfb_scalar_reactivities_are_positive():
    """Expected: ENDF/B-VIII.0 tabulated scalar reactivities are finite and non-negative."""
    temp_keV = 10.0

    dt = float(sigmav_DT_ENDFB_VIII0(temp_keV))
    dd_total = float(sigmav_DD_ENDFB_VIII0(temp_keV))
    ddn = float(sigmav_DDn_ENDFB_VIII0(temp_keV))
    ddp = float(sigmav_DDp_ENDFB_VIII0(temp_keV))
    dhe3 = float(sigmav_DHe3_ENDFB_VIII0(temp_keV))
    tt = float(sigmav_TT_ENDFB_VIII0(temp_keV))
    he3he3 = float(sigmav_He3He3_ENDFB_VIII0(temp_keV))
    the3_np = float(sigmav_THe3_np_ENDFB_VIII0(temp_keV))
    the3_d = float(sigmav_THe3_D_ENDFB_VIII0(temp_keV))
    the3_total = float(sigmav_THe3_ENDFB_VIII0(temp_keV))

    assert math.isfinite(dt) and dt > 0.0
    assert math.isfinite(dd_total) and dd_total > 0.0
    assert math.isfinite(ddn) and ddn > 0.0
    assert math.isfinite(ddp) and ddp > 0.0
    assert math.isclose(dd_total, ddn + ddp, rel_tol=1e-12)
    assert math.isfinite(dhe3) and dhe3 > 0.0
    assert math.isfinite(tt) and tt > 0.0
    assert math.isfinite(he3he3) and he3he3 > 0.0
    assert math.isfinite(the3_np) and the3_np > 0.0
    assert math.isfinite(the3_d) and the3_d > 0.0
    assert math.isclose(the3_total, the3_np + the3_d, rel_tol=1e-12)


def test_dd_total_relations_match_branch_and_table_helpers():
    """Expected: public DD total relations agree with branch helpers and NRL wrapper."""
    temp_keV = np.asarray([5.0, 10.0, 20.0], dtype=float)

    dd_bh_total = np.asarray(sigmav_DDn_BoschHale(temp_keV), dtype=float) + np.asarray(sigmav_DDp_BoschHale(temp_keV), dtype=float)
    dd_hively_total = np.asarray(sigmav_DDn_Hively(temp_keV), dtype=float) + np.asarray(sigmav_DDp_Hively(temp_keV), dtype=float)
    dd_endfb0_total = np.asarray(sigmav_DDn_ENDFB_VIII0(temp_keV), dtype=float) + np.asarray(sigmav_DDp_ENDFB_VIII0(temp_keV), dtype=float)

    assert np.allclose(np.asarray(sigmav_DD_BoschHale(temp_keV), dtype=float), dd_bh_total)
    assert np.allclose(np.asarray(sigmav_DD_Hively(temp_keV), dtype=float), dd_hively_total)
    assert np.allclose(np.asarray(sigmav_DD_ENDFB_VIII0(temp_keV), dtype=float), dd_endfb0_total)
    assert np.allclose(
        np.asarray(sigmav_DD_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_DD_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )


def test_endfb_viii1_scalar_reactivities_are_positive():
    """Expected: ENDF/B-VIII.1 tabulated scalar reactivities are finite and non-negative."""
    temp_keV = 20.0

    dt = float(sigmav_DT_ENDFB_VIII1(temp_keV))
    dd_total = float(sigmav_DD_ENDFB_VIII1(temp_keV))
    ddn = float(sigmav_DDn_ENDFB_VIII1(temp_keV))
    ddp = float(sigmav_DDp_ENDFB_VIII1(temp_keV))
    dhe3 = float(sigmav_DHe3_ENDFB_VIII1(temp_keV))
    tt = float(sigmav_TT_ENDFB_VIII1(temp_keV))
    he3he3 = float(sigmav_He3He3_ENDFB_VIII1(temp_keV))
    the3_np = float(sigmav_THe3_np_ENDFB_VIII1(temp_keV))
    the3_d = float(sigmav_THe3_D_ENDFB_VIII1(temp_keV))
    the3_total = float(sigmav_THe3_ENDFB_VIII1(temp_keV))

    assert math.isfinite(dt) and dt > 0.0
    assert math.isfinite(dd_total) and dd_total > 0.0
    assert math.isfinite(ddn) and ddn > 0.0
    assert math.isfinite(ddp) and ddp > 0.0
    assert math.isclose(dd_total, ddn + ddp, rel_tol=1e-12)
    assert math.isfinite(dhe3) and dhe3 > 0.0
    assert math.isfinite(tt) and tt > 0.0
    assert math.isfinite(he3he3) and he3he3 >= 0.0
    assert math.isfinite(the3_np) and the3_np >= 0.0
    assert math.isfinite(the3_d) and the3_d >= 0.0
    assert math.isclose(the3_total, the3_np + the3_d, rel_tol=1e-12)


def test_endfb_array_reactivities_preserve_shape_and_branch_sums():
    """Expected: ENDF/B-VIII.0 array helpers preserve shape and branch sums."""
    temp_keV = np.asarray([5.0, 10.0, 20.0], dtype=float)

    dt = np.asarray(sigmav_DT_ENDFB_VIII0(temp_keV), dtype=float)
    dd_total = np.asarray(sigmav_DD_ENDFB_VIII0(temp_keV), dtype=float)
    ddn = np.asarray(sigmav_DDn_ENDFB_VIII0(temp_keV), dtype=float)
    ddp = np.asarray(sigmav_DDp_ENDFB_VIII0(temp_keV), dtype=float)
    dhe3 = np.asarray(sigmav_DHe3_ENDFB_VIII0(temp_keV), dtype=float)
    tt = np.asarray(sigmav_TT_ENDFB_VIII0(temp_keV), dtype=float)
    he3he3 = np.asarray(sigmav_He3He3_ENDFB_VIII0(temp_keV), dtype=float)
    the3_total = np.asarray(sigmav_THe3_ENDFB_VIII0(temp_keV), dtype=float)
    the3_np = np.asarray(sigmav_THe3_np_ENDFB_VIII0(temp_keV), dtype=float)
    the3_d = np.asarray(sigmav_THe3_D_ENDFB_VIII0(temp_keV), dtype=float)

    assert dt.shape == temp_keV.shape
    assert dd_total.shape == temp_keV.shape
    assert ddn.shape == temp_keV.shape
    assert ddp.shape == temp_keV.shape
    assert dhe3.shape == temp_keV.shape
    assert tt.shape == temp_keV.shape
    assert he3he3.shape == temp_keV.shape
    assert the3_total.shape == temp_keV.shape
    assert np.all(dt > 0.0)
    assert np.all(dd_total > 0.0)
    assert np.all(dhe3 > 0.0)
    assert np.all(tt > 0.0)
    assert np.all(he3he3 > 0.0)
    assert np.allclose(dd_total, ddn + ddp)
    assert np.allclose(the3_total, the3_np + the3_d)


def test_endfb_reactivity_relations_expose_symbolic_models():
    """Expected: ENDF/B-VIII.0 reactivity relations also register symbolic expressions."""
    assert sigmav_DD_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_DT_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_DDn_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_DDp_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_DHe3_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_TT_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_He3He3_ENDFB_VIII0.sympy_expression is not None
    assert sigmav_THe3_ENDFB_VIII0.sympy_expression is not None


def test_endfb_viii1_reactivity_relations_expose_symbolic_models():
    """Expected: ENDF/B-VIII.1 tabulated reactivity relations also register symbolic expressions."""
    assert sigmav_DD_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_DT_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_DDn_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_DDp_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_DHe3_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_TT_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_He3He3_ENDFB_VIII1.sympy_expression is not None
    assert sigmav_THe3_ENDFB_VIII1.sympy_expression is not None


def test_nrl_reactivity_relations_expose_symbolic_models_without_numeric_rows():
    """Expected: NRL tabulated reactivity relations register symbolic expressions before tables are populated."""
    assert sigmav_DD_NRL.sympy_expression is not None
    assert sigmav_DT_NRL.sympy_expression is not None
    assert sigmav_DHe3_NRL.sympy_expression is not None
    assert sigmav_TT_NRL.sympy_expression is not None
    assert sigmav_THe3_NRL.sympy_expression is not None


def test_nrl_reactivities_default_to_pchip_interpolation():
    """Expected: NRL helper defaults match explicitly requested PCHIP interpolation."""
    temp_keV = np.asarray([3.0, 30.0, 300.0], dtype=float)

    assert np.allclose(
        np.asarray(sigmav_DT_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_DT_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
    )
    assert np.allclose(
        np.asarray(sigmav_DD_NRL(temp_keV), dtype=float),
        np.asarray(sigmav_DD_NRL(temp_keV, interpolation_kind="pchip"), dtype=float),
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


def test_reactivity_relations_have_method_specific_names():
    """Expected: each reactivity relation name encodes the selectable method."""
    assert sigmav_DT_BoschHale.name == "DT reactivity BoschHale"
    assert sigmav_DT_Hively.name == "DT reactivity Hively"
    assert sigmav_DT_ENDFB_VIII0.name == "DT reactivity ENDFB-VIII0"
    assert sigmav_DT_NRL.name == "DT reactivity NRL"
    assert sigmav_DDn_BoschHale.name == "DDn reactivity BoschHale"
    assert sigmav_DDn_Hively.name == "DDn reactivity Hively"
    assert sigmav_DDn_ENDFB_VIII0.name == "DDn reactivity ENDFB-VIII0"
    assert sigmav_DDp_BoschHale.name == "DDp reactivity BoschHale"
    assert sigmav_DDp_Hively.name == "DDp reactivity Hively"
    assert sigmav_DDp_ENDFB_VIII0.name == "DDp reactivity ENDFB-VIII0"
    assert sigmav_DHe3_BoschHale.name == "DHe3 reactivity BoschHale"
    assert sigmav_DHe3_ENDFB_VIII0.name == "DHe3 reactivity ENDFB-VIII0"
    assert sigmav_DHe3_NRL.name == "DHe3 reactivity NRL"
    assert sigmav_TT_CF88.name == "TT reactivity CF88"
    assert sigmav_TT_ENDFB_VIII0.name == "TT reactivity ENDFB-VIII0"
    assert sigmav_TT_NRL.name == "TT reactivity NRL"
    assert sigmav_He3He3_CF88.name == "He3He3 reactivity CF88"
    assert sigmav_He3He3_ENDFB_VIII0.name == "He3He3 reactivity ENDFB-VIII0"
    assert sigmav_THe3_D_CF88.name == "THe3_D reactivity CF88"
    assert sigmav_THe3_D_ENDFB_VIII0.name == "THe3_D reactivity ENDFB-VIII0"
    assert sigmav_THe3_D_ENDFB_VIII1.name == "THe3_D reactivity ENDFB-VIII1"
    assert sigmav_THe3_np_CF88.name == "THe3_np reactivity CF88"
    assert sigmav_THe3_np_ENDFB_VIII0.name == "THe3_np reactivity ENDFB-VIII0"
    assert sigmav_THe3_np_ENDFB_VIII1.name == "THe3_np reactivity ENDFB-VIII1"
    assert sigmav_THe3_CF88.name == "THe3 reactivity CF88"
    assert sigmav_THe3_ENDFB_VIII0.name == "THe3 reactivity ENDFB-VIII0"
    assert sigmav_THe3_NRL.name == "THe3 reactivity NRL"


def test_the3_branch_reactivity_outputs_are_registered():
    """Expected: branch-level T-He3 reactivities are exposed as selectable relation outputs."""
    assert sigmav_THe3_D_CF88.preferred_target == "sigmav_THe3_D"
    assert sigmav_THe3_np_CF88.preferred_target == "sigmav_THe3_np"
    assert allowed_variable_ndim("sigmav_THe3_D") == 1
    assert allowed_variable_ndim("sigmav_THe3_np") == 1


def test_reactor_defaults_assign_reactivity_methods_without_overwriting_user_choice():
    """Expected: reactor defaults create reactivity variables with default methods but preserve explicit overrides."""
    variables = {
        "sigmav_DT": make_variable(
            name="sigmav_DT",
            ndim=1,
            method="custom dt method",
        ),
    }

    apply_reactor_defaults(variables)

    assert variables["sigmav_DT"].method == "custom dt method"
    assert variables["sigmav_DDn"].method == "DDn reactivity BoschHale"
    assert variables["sigmav_DDp"].method == "DDp reactivity BoschHale"
    assert variables["sigmav_DHe3"].method == "DHe3 reactivity BoschHale"
    assert variables["sigmav_TT"].method == "TT reactivity CF88"
    assert variables["sigmav_He3He3"].method == "He3He3 reactivity CF88"
    assert variables["sigmav_THe3"].method == "THe3 reactivity CF88"


def test_removed_reactivity_profile_variable_aliases_are_not_registered():
    """Expected: legacy reactivity *_profile variables are no longer part of the registry."""
    assert allowed_variable_ndim("sigmav_DT_profile") == 0
    assert allowed_variable_ndim("sigmav_DDn_profile") == 0
    assert allowed_variable_ndim("sigmav_DDp_profile") == 0
    assert allowed_variable_ndim("sigmav_DHe3_profile") == 0
    assert allowed_variable_ndim("sigmav_TT_profile") == 0
    assert allowed_variable_ndim("sigmav_He3He3_profile") == 0
    assert allowed_variable_ndim("sigmav_THe3_profile") == 0


def test_method_override_filters_reactivity_relations_by_name():
    """Expected: method selection keeps the requested reactivity relation and rejects others with the same output."""
    rels = get_filtered_relations(
        ("fusion_power",),
        ("T_i", "sigmav_TT"),
        ("TT reactivity ENDFB-VIII0",),
    )
    names = {
        rel.name
        for rel in rels
        if rel.preferred_target == "sigmav_TT"
    }
    assert "TT reactivity ENDFB-VIII0" in names
    assert "TT reactivity" not in names
    assert "TT reactivity CF88" not in names
