"""HDF5 result archiving: one format for every mode."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from fusdb import RelationSystem, Variable, load_result, save_result
from fusdb.registry import RELATIONS


def _small_system() -> RelationSystem:
    return RelationSystem(
        [Variable("R", 3.0), Variable("a", 1.0), Variable("A")],
        [RELATIONS.get("aspect_ratio")],
    )


def test_reconcile_result_roundtrips_through_h5(tmp_path) -> None:
    system = _small_system()
    result = system.run("reconcile", save=tmp_path / "run.h5")

    loaded = load_result(tmp_path / "run.h5")

    assert loaded["mode"] == "reconcile"
    assert loaded["success"] == result["success"]
    assert loaded["termination"] == result["termination"]
    assert loaded["failed_relations"] == list(result["failed_relations"])
    assert loaded["max_residual"] == pytest.approx(result["max_residual"])
    assert set(loaded["relation_status"]) == set(result["relation_status"])
    np.testing.assert_allclose(loaded["values"]["A"], result["values"]["A"])


def test_result_is_plain_picklable_data(tmp_path) -> None:
    import pickle

    result = _small_system().run("reconcile")
    pickle.loads(pickle.dumps(result))
    assert "certificate" not in result
    assert "relations" not in result


def test_save_result_handles_popcon_payload_shapes(tmp_path) -> None:
    grid = np.arange(6, dtype=float).reshape(2, 3)
    payload = {
        "mode": "popcon",
        "success": True,
        "popcon": {
            "x": {"name": "n_e_avg", "values": np.asarray([1.0, 2.0, 3.0])},
            "fields": {"P_fus": grid},
            "success": grid > 2,
            "failures": [{"ix": 0, "iy": 1, "termination": "certification failed"}],
        },
        "warnings": [],
        "note": None,
    }
    save_result(payload, tmp_path / "scan.h5")
    loaded = load_result(tmp_path / "scan.h5")

    np.testing.assert_allclose(loaded["popcon"]["fields"]["P_fus"], grid)
    np.testing.assert_array_equal(loaded["popcon"]["success"], grid > 2)
    assert loaded["popcon"]["failures"][0]["termination"] == "certification failed"
    assert loaded["note"] is None
