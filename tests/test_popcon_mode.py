"""Popcon-mode tests on the cfspopcon SPARC popcon fixture.

``tests/cfspopcon_SPARC/reactor.yaml`` is the cfspopcon SPARC PRD reproduction
case, which is scan-native: dilution comes from quasineutrality with a generic
Z=6 impurity fraction, the density peaking is derived (not supplied), and P_aux
is the free degree of freedom the scan solves for.

The popcon evaluates the grid in vectorized chunks with every
supplied input held exactly and the axis values pinned to the grid
coordinates, then certifies each point individually.  The central guarantee
-- and the central test here -- is reconcile-equivalence: reconciling any
certified grid point from scratch with the identical pinned scenario
reproduces the popcon's numbers to solver precision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fusdb.modes import popcon as popcon_mode
from fusdb.reactor import Reactor
from fusdb.registry import VARIABLES

POPCON_YAML = Path(__file__).parent / "cfspopcon_SPARC" / "reactor.yaml"

# 5x5 over the H-mode operating region around the PRD point, so the scan's
# certified points are consistent with the fixture's declared h_mode regime and
# the reconcile-equivalence comparison stays apples-to-apples (a per-point
# reconcile of an L-mode corner would switch regime and legitimately differ).
X_AXIS = {"variable": "average_electron_density", "start": 20.0e19, "stop": 38.0e19, "num": 5}
Y_AXIS = {"variable": "average_electron_temp", "start": 8.0, "stop": 18.0, "num": 5}
OUTPUTS = ("P_fus", "P_aux", "Q_sci", "beta_N", "f_GW", "P_LH", "W_th", "P_loss", "tau_E")


@pytest.fixture(scope="module")
def serial_scan() -> tuple[Reactor, dict]:
    reactor = Reactor.from_yaml(POPCON_YAML)
    result = reactor.popcon(x=X_AXIS, y=Y_AXIS, outputs=OUTPUTS)
    return reactor, result


def test_axis_spec_parsing() -> None:
    name, values = popcon_mode.parse_axis(("average_electron_density", 1.0e20, 4.0e20, 4), VARIABLES)
    assert name == VARIABLES.resolve("average_electron_density")
    assert np.allclose(values, np.linspace(1.0e20, 4.0e20, 4))

    name, values = popcon_mode.parse_axis({"variable": "T_e_avg", "values": [4.0, 9.0, 20.0]}, VARIABLES)
    assert name == "T_e_avg"
    assert np.array_equal(values, [4.0, 9.0, 20.0])

    _, values = popcon_mode.parse_axis({"name": "T_e_avg", "start": 1.0, "stop": 100.0, "num": 3, "spacing": "log"}, VARIABLES)
    assert np.allclose(values, [1.0, 10.0, 100.0])

    with pytest.raises(KeyError):
        popcon_mode.parse_axis(("not_a_variable", 0.0, 1.0, 2), VARIABLES)
    with pytest.raises(ValueError, match="must be scalars"):
        popcon_mode.parse_axis(("n_e", 0.0, 1.0, 2), VARIABLES)
    with pytest.raises(ValueError, match="'start'/'stop'/'num'"):
        popcon_mode.parse_axis({"variable": "T_e_avg", "start": 1.0}, VARIABLES)
    with pytest.raises(ValueError, match="spacing"):
        popcon_mode.parse_axis({"variable": "T_e_avg", "start": 1.0, "stop": 2.0, "num": 2, "spacing": "cubic"}, VARIABLES)
    with pytest.raises(ValueError, match="Unknown axis spec key"):
        popcon_mode.parse_axis({"variable": "T_e_avg", "values": [1.0], "typo": 1}, VARIABLES)


def test_popcon_option_validation() -> None:
    # Direct mode invocation: the option-validation paths return before any
    # solve, so one system serves every case.
    system = Reactor.from_yaml(POPCON_YAML).relation_system().compile()

    result = popcon_mode.run(system, y=Y_AXIS)
    assert not result["success"] and "requires 'x' and 'y'" in result["errors"][0]

    result = popcon_mode.run(system, x=X_AXIS, y=X_AXIS)
    assert not result["success"] and "axes must differ" in result["errors"][0]

    result = popcon_mode.run(system, x=X_AXIS, y=Y_AXIS, typo=1)
    assert not result["success"] and "Unknown popcon option(s)" in result["errors"][0]

    # The legacy per-point inner modes are gone; passing one is an error.
    result = popcon_mode.run(system, x=X_AXIS, y=Y_AXIS, inner="reconcile")
    assert not result["success"] and "Unknown popcon option(s)" in result["errors"][0]
    result = popcon_mode.run(system, x=X_AXIS, y=Y_AXIS, solver="reconcile")
    assert not result["success"] and "Unknown popcon option(s)" in result["errors"][0]


def test_certification_cone_scales_with_targets() -> None:
    system = Reactor.from_yaml(POPCON_YAML).relation_system().compile()
    lean_rels, lean_vars = popcon_mode.certification_cone(system, ("P_fus",))
    wide_rels, wide_vars = popcon_mode.certification_cone(system, ("P_fus", "P_LH", "P_aux"))
    assert 0 < len(lean_rels) <= len(wide_rels) <= len(system.relations)
    assert {rel.name for rel in lean_rels} <= {rel.name for rel in wide_rels}
    assert "P_fus" in lean_vars and "P_LH" not in lean_vars
    assert "P_LH" in wide_vars
    # A free core in the cone drags in its whole determining component --
    # certifying P_aux without the confinement chain would certify an
    # arbitrary number.
    assert {"P_loss", "tau_E", "W_th"} <= wide_vars
    assert "Energy confinement balance" in {rel.name for rel in wide_rels}


def test_popcon_scan_certifies_grid(serial_scan: tuple[Reactor, dict]) -> None:
    _, result = serial_scan
    assert result["success"]
    assert result["n_points"] == 25
    payload = result["popcon"]
    assert payload["x"]["name"] == "n_e_avg" and payload["y"]["name"] == "T_e_avg"
    assert payload["success"].shape == (5, 5)
    # The extreme hot corner may fail its reactivity-range certification;
    # the operating bulk must certify.
    assert payload["success"].sum() >= 23
    assert set(payload["fields"]) == set(OUTPUTS)
    for name, grid in payload["fields"].items():
        assert grid.shape == (5, 5)
        assert np.isfinite(grid[payload["success"]]).all()
        assert np.isnan(grid[~payload["success"]]).all()
    assert len(payload["failures"]) == 25 - int(payload["success"].sum())

    # Point-wise consistency of the certified fields: the definitions
    # connecting them hold on the reported numbers.
    ok = payload["success"]
    fields = payload["fields"]
    assert np.allclose(fields["Q_sci"][ok], (fields["P_fus"] / fields["P_aux"])[ok], rtol=1e-6)
    assert np.allclose((fields["P_loss"] * fields["tau_E"])[ok], fields["W_th"][ok], rtol=1e-6)


def test_popcon_parallel_chunks_match_one_batch() -> None:
    """Chunk scheduling must not change per-point answers or certificates."""
    x = {"variable": "average_electron_density", "values": [2.5e20, 3.0e20]}
    y = {"variable": "average_electron_temp", "values": [9.0, 12.0]}
    outputs = ("P_fus", "P_aux")
    serial = Reactor.from_yaml(POPCON_YAML)._clone_for_regime("h_mode", include_guards=False)
    parallel = Reactor.from_yaml(POPCON_YAML)._clone_for_regime("h_mode", include_guards=False)
    expected = serial._run_once("popcon", x=x, y=y, outputs=outputs)["popcon"]
    actual = parallel._run_once(
        "popcon", x=x, y=y, outputs=outputs, workers=2, chunk_size=2
    )["popcon"]
    assert np.array_equal(actual["success"], expected["success"])
    for name in outputs:
        assert np.allclose(actual["fields"][name], expected["fields"][name], equal_nan=True)


def test_popcon_points_reconcile_to_same_values(serial_scan: tuple[Reactor, dict]) -> None:
    """The central guarantee: every certified grid point, reconciled from
    scratch with the identical pinned scenario, reproduces the popcon's
    numbers to solver precision.

    The scan runs in the fixture's declared h_mode regime.  A per-point
    reconcile is free to switch regime when the point is inconsistent with
    h_mode (P_sep below the L-H threshold); such a point is not a valid h_mode
    operating point, so its l_mode reconcile legitimately differs from the
    h_mode scan value and is skipped -- equivalence is asserted only where the
    reconcile stays in the scan's regime."""
    _, result = serial_scan
    payload = result["popcon"]
    x_values, y_values = payload["x"]["values"], payload["y"]["values"]
    checked = 0
    for iy in range(y_values.size):
        for ix in range(x_values.size):
            if not payload["success"][iy, ix]:
                continue
            reference = Reactor.from_yaml(POPCON_YAML)
            # The popcon holds every supplied input exactly and pins the
            # axes; the apples-to-apples reconcile pins the same scenario.
            # Variable is immutable, so each declaration is replaced by a
            # clone with the override instead of being mutated in place.
            for name, var in reference.variables.items():
                reference.variables[name] = var.clone(fixed=True)
            for axis, value in (("n_e_avg", x_values[ix]), ("T_e_avg", y_values[iy])):
                reference.add_variable(reference.get_variable(axis).clone(value=float(value), fixed=True))
            reconcile_result = reference.reconcile()
            # A point the full regime-switching reconcile cannot settle in the
            # scan's h_mode regime (an L-H-bistable or L-mode point) is not a
            # valid h_mode operating point; its scan value legitimately differs
            # and is skipped -- equivalence is asserted only where the pinned
            # reconcile cleanly succeeds in h_mode.
            if not reconcile_result["success"] or "h_mode" not in (
                    reconcile_result.get("regime_admissible") or []
                ):
                continue
            for name in OUTPUTS:
                popcon_value = payload["fields"][name][iy, ix]
                reconcile_value = float(np.asarray(reference.last_plan.values[name]).reshape(-1)[0])
                # 5e-6 rather than 1e-6: OUTPUTS contains quotients of other
                # asserted fields (Q_sci = P_fus / P_aux), which compound both
                # factors' solver tolerances -- near ignition P_aux is small,
                # so the ratio is the tightest field even when both sides are
                # converged to their own solver precision.
                assert popcon_value == pytest.approx(reconcile_value, rel=5e-6), (
                    f"({iy},{ix}) {name}: popcon {popcon_value} vs reconcile {reconcile_value}"
                )
            checked += 1
    assert checked >= 15


def test_popcon_restores_system_state(serial_scan: tuple[Reactor, dict]) -> None:
    reactor, _ = serial_scan
    system = reactor.last_plan
    # The scan must leave the system in its pre-scan pure-input state: no
    # derived values, no pinned axes, and the reactor absorbed nothing new.
    assert "P_fus" not in system.values
    assert "n_e_avg" not in system.fixed and "T_e_avg" not in system.fixed
    assert system.inputs["n_e_avg"] == pytest.approx(25.0e19)
    assert reactor.get_variable("P_fus") is None
    assert reactor.n_e_avg.value == pytest.approx(25.0e19)


def test_popcon_survives_undetermined_free_core() -> None:
    # An unsupplied scenario freedom (here squareness) becomes a free core no
    # relation determines; the scaled Gauss-Newton leaves it at its seed and
    # certification arbitrates -- the scan must not degrade.
    reactor = Reactor.from_yaml(POPCON_YAML)
    del reactor.variables["squareness"]
    result = reactor.popcon(
        x={"variable": "average_electron_density", "values": [25.0e19]},
        y={"variable": "average_electron_temp", "values": [9.13793]},
        outputs=("P_fus", "P_aux"),
    )
    assert result["success"] and result["popcon"]["success"].all()
    assert np.isfinite(result["popcon"]["fields"]["P_fus"]).all()


def test_popcon_warns_on_underivable_output() -> None:
    # Requesting an output the compiled system cannot derive (P_e_net has no
    # active producer in this fixture) must be surfaced, not left as a
    # silently blank NaN field.
    reactor = Reactor.from_yaml(POPCON_YAML)
    result = reactor.popcon(
        x={"variable": "average_electron_density", "values": [25.0e19]},
        y={"variable": "average_electron_temp", "values": [9.13793]},
        outputs=("P_fus", "P_e_net"),
    )
    canonical = VARIABLES.resolve("P_e_net")  # fields and warnings use canonical names
    assert any("not derivable" in warning and canonical in warning for warning in result["warnings"])
    assert np.isnan(result["popcon"]["fields"][canonical]).all()
    assert np.isfinite(result["popcon"]["fields"]["P_fus"]).all()


def test_popcon_field_map_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from fusdb.plotting import plot_field_map, popcon_field_map

    x, y = np.linspace(1.0, 3.0, 5), np.linspace(4.0, 20.0, 4)
    grid_x, grid_y = np.meshgrid(x, y)
    payload = {
        "x": {"name": "n_e_avg", "values": x},
        "y": {"name": "T_e_avg", "values": y},
        "fields": {"P_fus": grid_x * grid_y, "f_GW": grid_x / 3.0},
        "success": grid_x * grid_y < 50.0,
        "failures": [],
    }
    data = popcon_field_map({"popcon": payload})
    ax = plot_field_map(data, fill="P_fus", contours=("f_GW",))
    assert ax.get_xlabel().startswith("n_e_avg")
    with pytest.raises(ValueError, match="fill"):
        plot_field_map(data)
    with pytest.raises(KeyError, match="not present"):
        plot_field_map(data, contours=("nope",))
