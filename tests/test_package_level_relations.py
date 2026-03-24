import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fusdb
from fusdb.relation_class import Relation


def test_dt_relation_is_exposed_at_package_level():
    """Expected: DT fusion power relation is importable as fusdb.fusion_power_dt."""
    assert hasattr(fusdb, "fusion_power_dt")
    assert isinstance(fusdb.fusion_power_dt, Relation)
    assert fusdb.fusion_power_dt.outputs == ("P_fus_DT",)
    assert fusdb.fusion_power_dt.evaluate(
        {"P_fus_DT_alpha": 10.0, "P_fus_DT_n": 40.0}
    ) == pytest.approx(50.0)


def test_non_relation_helpers_are_not_exposed_at_package_level():
    """Expected: relation helper functions stay off the top-level fusdb namespace."""
    assert not hasattr(fusdb, "load_table")
    assert not hasattr(fusdb, "tau_E_ipb98y")
