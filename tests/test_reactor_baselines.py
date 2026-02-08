from __future__ import annotations

import json
import math
from pathlib import Path

from fusdb.reactor_class import Reactor
from fusdb.variable_class import Variable


BASELINE_PATH = Path(__file__).parent / "data" / "reactor_baselines.json"


def _numeric_value(value: object) -> float | None:
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import sympy as sp
    except Exception:
        sp = None
    if np is not None and isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if sp is not None and isinstance(value, sp.Expr):
        try:
            return float(value.evalf())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _assert_close(actual: float, expected: float, *, rel: float = 1e-6, abs_tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(f"expected {expected}, got {actual} (rel={rel}, abs={abs_tol})")


def test_reactor_baselines() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())
    for rid, expected_map in baseline.items():
        reactor = Reactor.from_yaml(f"reactors/{rid}")
        reactor.solve()
        for name, expected in expected_map.items():
            raw = Variable.get_from_dict(
                reactor.variables_dict,
                name,
                allow_override=True,
                mode="current",
            )
            actual = _numeric_value(raw)
            assert actual is not None and math.isfinite(actual), f"{rid}:{name} non-numeric -> {raw}"
            _assert_close(actual, expected)
