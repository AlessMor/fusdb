from pathlib import Path

import pytest

from fusdb import Reactor


STELLARIS = Path(__file__).parents[1] / "reactors" / "STELLARIS" / "reactor.yaml"


def test_stellaris_reconcile_stays_on_pre_coordinate_refactor_design_point():
    """Guard the behavior-neutral stellarator repair against silent pruning.

    These are the pre-coordinate-refactor values measured for the STELLARIS
    scenario. The reduced ``w_V=rho`` and straight-rho ``n_la`` providers are
    intentionally compatibility models, so introducing explicit coordinate
    plumbing must not move this operating point. A future equilibrium-correct
    stellarator measure should update this test only as an isolated physics
    change with its own quantified comparison.
    """
    result = Reactor.from_yaml(STELLARIS).run("reconcile")
    assert result["success"] is True
    values = result["values"]

    assert values["tau_p"] == pytest.approx(10.989, rel=5e-3)
    assert values["tau_E"] == pytest.approx(1.0465, rel=5e-3)
    assert values["T_e_avg"] == pytest.approx(9.8417, rel=5e-3)
    assert values["n_e_avg"] == pytest.approx(3.180e20, rel=5e-3)
    assert len(result.get("inputs_beyond_tolerance") or ()) <= 2
