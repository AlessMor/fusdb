import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.utils import as_profile_array, mean_profile, scalarize_mapping, scalarize_value


def test_as_profile_array_and_mean_profile() -> None:
    """Expected: reducers return profile arrays and deterministic means."""
    arr = np.asarray([1.0, 2.0, 3.0], dtype=float)
    assert np.array_equal(as_profile_array(arr), arr)
    assert mean_profile(arr) == 2.0
    assert as_profile_array(1.0) is None
    assert mean_profile(1.0) is None


def test_scalarize_value_prefers_profile_mean_then_float() -> None:
    """Expected: scalarize_value returns profile mean or finite scalar when possible."""
    assert scalarize_value(np.asarray([2.0, 4.0], dtype=float)) == 3.0
    assert scalarize_value(5.0) == 5.0


def test_scalarize_mapping_applies_ndim_lookup() -> None:
    """Expected: scalarize_mapping only reduces ndim=1 entries."""
    values = {"a": np.asarray([1.0, 3.0], dtype=float), "b": 2.0}

    def _ndim(name: str) -> int:
        return 1 if name == "a" else 0

    out = scalarize_mapping(values, ndim_lookup=_ndim)
    assert out["a"] == 2.0
    assert out["b"] == 2.0
