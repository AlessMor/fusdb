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

    assert rel(P_loss=150.0, P_charged=120.0, P_aux=30.0) is True
    assert rel(P_loss=149.0, P_charged=120.0, P_aux=30.0) is False
