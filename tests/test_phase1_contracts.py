"""Phase-1 behavior contracts: exact reconcile, diagnostics, recorded causes.

Covers the S8a ``exact`` option (movement deadzone removal), the S10a
diagnostics channel (first recorded causes behind the 1e12 residual barrier
and skipped completion providers), and their presence in reconcile results.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fusdb import Reactor
from fusdb.variable import Variable

DEMO_YAML = Path(__file__).parents[1] / "reactors" / "DEMO_2022" / "reactor.yaml"
POLOMAC_YAML = Path(__file__).parents[1] / "reactors" / "Polomac" / "reactor.yaml"
ARC_YAML = Path(__file__).parents[1] / "reactors" / "ARC_V0" / "reactor.yaml"
SPARC_YAML = Path(__file__).parent / "cfspopcon_SPARC" / "reactor.yaml"


def _demo_system():
    reactor = Reactor.from_yaml(DEMO_YAML)
    candidate = reactor._clone_for_regime("h_mode", include_guards=False)
    system = candidate.relation_system()
    system.compile()
    return system


def test_movement_rows_deadzone_flag():
    """deadzone=True gives free movement inside tolerance; exact penalises it."""
    system = _demo_system()
    system._movement_plan = [("x_test", 1.0, 0.1, True, None)]
    inside_band = {"x_test": 1.05}
    assert dict(system._movement_rows(inside_band))["x_test"] == 0.0
    exact_excess = dict(system._movement_rows(inside_band, deadzone=False))["x_test"]
    assert abs(exact_excess - 0.5) < 1e-12


def test_reconcile_result_carries_diagnostics():
    system = _demo_system()
    result = system.run("reconcile")
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics["residual_failures"], list)
    assert isinstance(diagnostics["completion_errors"], dict)


def test_exact_reconcile_runs():
    system = _demo_system()
    result = system.run("reconcile", exact=True)
    assert "diagnostics" in result
    assert "inputs_beyond_tolerance" in result
    assert result["termination"]


def test_completion_provider_failure_is_recorded():
    """A raising provider is skipped, but its first cause survives (S10a)."""
    system = _demo_system()
    plan = system._provider_plan
    assert plan, "DEMO compile produced no completion providers"
    rel = plan[0][0]
    values = system.solver_values()
    with patch.object(rel, "evaluate", side_effect=RuntimeError("boom")):
        system.complete(values)
    assert rel.name in system.completion_errors
    assert "boom" in system.completion_errors[rel.name]


def test_reported_roles_origins():
    """One role per variable: inactive / fixed / movable / computed / assumed."""
    s = _demo_system()
    s.pack()
    rel = s.reported_roles()
    assert rel.get("P_fus") == "movable"         # supplied, may move in tolerance
    assert rel.get("kappa") == "computed"        # the equations determine it
    assert rel.get("f_He4") == "assumed"         # registry constant, nothing pins it
    assert "inactive" not in set(rel.values())   # inactive never reported
    assert "rho" not in rel                      # coordinate excluded
    # Informative, not a smear: more than one origin present.
    assert len(set(rel.values())) >= 3
    # avg_to_profile is the orthogonal flag, not a role.
    assert {"T_e", "n_e", "T_i"} <= s.avg_to_profile
    assert s.avg_to_profile <= set(rel)
    assert not (s.avg_to_profile & {"P_fus", "kappa"})


def test_polomac_profiles_are_reconstructed_not_undetermined():
    """Polomac reconstructs its profiles instead of leaving their levels free.

    Polomac declares only the tag ``polomac``, which is in no ``device`` group.
    The fallback profile generators used to enumerate (tokamak, stellarator,
    mirror), so none of them matched: Polomac had no 1-D profiles at all, every
    consumer of them was pruned, and what survived ran on whatever producer
    happened to need only averages -- with ``T_i_avg`` floating free to 50 keV on
    a 100 eV device.  With the generators device-agnostic, the profiles are a
    uniform shape at the average value, so S9 reports nothing under-determined.
    """
    s = Reactor.from_yaml(POLOMAC_YAML).relation_system()
    s.compile()
    s.pack()
    rel = s.reported_roles()
    for prof in ("n_e", "T_e", "T_i"):
        # Computed from the averages, and flagged as expanded with an assumed
        # shape -- not "assumed", whose level nothing would pin.
        assert rel.get(prof) == "computed", (prof, rel.get(prof))
        assert prof in s.avg_to_profile, prof
    assert list(s.underdetermined_profiles or ()) == []
    # The composition defaults are held constants here (no tau_p to derive them).
    assert rel.get("f_He4") == "assumed"


def test_polomac_uses_default_producers_and_is_physical():
    """Polomac runs fusdb's own default producers and reproduces hand values."""
    result = Reactor.from_yaml(POLOMAC_YAML).run("reconcile")
    assert result["success"] is True
    values = result["values"]
    # p_th = (n_e T_e + n_i T_i) = 2 * 1e20 * 100 eV; W_th = 3/2 p_th V_p.
    assert values["p_th"] == pytest.approx(3204.353268, rel=1e-9)
    assert values["W_th"] == pytest.approx(1.5 * 3204.353268 * 0.15, rel=1e-9)
    assert values["T_i_avg"] == pytest.approx(0.1, rel=1e-9)   # not a free 50 keV
    assert values["Z_eff"] == pytest.approx(1.0, rel=1e-12)    # f_He4 = 0, no c_z


def test_reconcile_diagnostics_consolidated():
    """The reconcile diagnostics block folds all signals into one place (D2)."""
    result = Reactor.from_yaml(DEMO_YAML).run("reconcile")
    diag = result["diagnostics"]
    assert set(diag) == {"residual_failures", "completion_errors", "underdetermined_profiles", "role_summary"}
    assert sum(diag["role_summary"].values()) > 0


def test_optimize_constraints_and_seed_options():
    """optimize accepts per-run constraints (abs + var-vs-var) and a warm start (F1)."""
    s = _demo_system()
    # invalid constraint op is rejected as a bad option, not silently ignored
    bad = s.run("optimize", objective="P_fus", sense="maximize", maxiter=1,
                constraints=[("P_fus", "!!", 1.0)])
    assert bad["termination"] == "invalid options"
    # valid options (constraints incl. a variable bound, initial_guesses) are accepted
    ok = s.run("optimize", objective="P_fus", sense="maximize", maxiter=1,
               constraints=[("P_fus", ">=", 2e9), ("P_in", ">=", "P_LH")],
               initial_guesses={"T_e_avg": 13.0})
    assert ok["mode"] == "optimize" and "invalid" not in ok["termination"]


def test_run_many_carries_every_mode_result():
    """`run_many` is faithful for scalar modes and keeps popcon's payload.

    The worker used to snapshot ``last_system.last_result``, which popcon never
    writes -- so a POPCON case silently returned empty columns.  Columns now
    carry whatever that mode's ``run`` returned.
    """
    import fusdb
    from fusdb.modes import MODE_NAMES

    arc = str(ARC_YAML)
    for mode in ("verify", "ordered", "reconcile"):
        column = fusdb.run_many(arc, [{"delta": 0.3}], mode=mode, workers=1)[0]
        reactor = Reactor.from_yaml(arc)
        reactor.add_variable(Variable("delta", value=0.3, fixed=True))
        direct = reactor.run(mode)
        assert set(column.result) == set(direct), mode
        assert column.result.get("success") == direct.get("success"), mode

    optimized = fusdb.run_many(
        arc,
        [{"delta": 0.3}],
        mode="optimize",
        workers=1,
        objective="P_fus",
        sense="maximize",
        maxiter=1,
    )[0]
    assert optimized.result["mode"] == "optimize"

    # popcon: one scan per case, payload intact.
    column = fusdb.run_many(
        str(SPARC_YAML), [{"B0": 12.2}], mode="popcon", workers=1,
        x={"variable": "average_electron_temp", "start": 8.0, "stop": 12.0, "num": 2},
        y={"variable": "average_electron_density", "start": 2.5e20, "stop": 3.0e20, "num": 2},
    )[0]
    assert column.result["success"] is True
    assert column.result["n_points"] == 4
    assert column.result["popcon"]["fields"]

    # every registered mode is reachable through run_many
    assert MODE_NAMES == {"verify", "ordered", "reconcile", "optimize", "popcon"}
