from __future__ import annotations

import pytest

from fusdb.registry import RELATIONS


def test_plasma_loss_power_solves_any_single_missing_variable():
    rel = RELATIONS.get("Plasma loss power")

    assert rel(P_charged=120.0, P_aux=30.0) == pytest.approx(150.0)
    assert rel(P_loss=150.0, P_charged=120.0) == pytest.approx(30.0)
    assert rel(P_loss=150.0, P_aux=30.0) == pytest.approx(120.0)


def test_plasma_loss_power_verifies_when_all_variables_are_supplied():
    rel = RELATIONS.get("Plasma loss power")

    # Reactor-scale watts: P_loss carries abs_tol=1e6 W, so violations must
    # exceed max(abs_tol, rel_tol * scale) to fail verification.
    assert rel(P_loss=150.0e6, P_charged=120.0e6, P_aux=30.0e6) is True
    assert rel(P_loss=140.0e6, P_charged=120.0e6, P_aux=30.0e6) is False
