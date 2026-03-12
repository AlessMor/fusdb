from __future__ import annotations

import json
import math
import numbers
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.reactor_class import Reactor
from fusdb.utils import as_profile_array


BASELINE_PATH = Path(__file__).parent / "data" / "reactor_baselines.json"
REACTORS_DIR = ROOT / "reactors"
REL_TOL = 1e-2
ABS_TOL = 1e-8


def _numeric_value(value: object) -> float | None:
    profile = as_profile_array(value)
    if profile is not None:
        return float(profile.mean())

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)

    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            return None
        if isinstance(value, np.generic):
            try:
                return float(value.item())
            except Exception:
                return None

    try:
        import sympy as sp
    except Exception:
        sp = None

    if sp is not None and isinstance(value, sp.Expr):
        try:
            return float(value.evalf())
        except Exception:
            return None

    try:
        return float(value)
    except Exception:
        return None


def _scan_reactor_paths() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in REACTORS_DIR.glob("*/reactor.yaml"):
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        metadata = raw.get("metadata", {}) or {}
        rid = metadata.get("id") or path.parent.name
        if rid in mapping:
            raise AssertionError(f"Duplicate reactor id '{rid}' at {path} and {mapping[rid]}")
        mapping[rid] = path
    return mapping


def _build_actual_map(reactor: Reactor) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name, var in reactor.variables_dict.items():
        values[name] = _numeric_value(var.current_value)
    return values


def test_reactor_baselines() -> None:
    """Expected: each reactor solves to numeric values matching stored baselines within configured tolerances."""
    baseline = json.loads(BASELINE_PATH.read_text())
    path_map = _scan_reactor_paths()

    for rid, expected_map in baseline.items():
        path = path_map.get(rid)
        assert path is not None, f"Missing reactor.yaml for id '{rid}'"

        reactor = Reactor.from_yaml(path)
        reactor.solve()
        actual_map = _build_actual_map(reactor)

        missing: list[str] = []
        max_dev: tuple[float, str, float, float] | None = None

        for name, expected in expected_map.items():
            if expected is None:
                continue
            actual = actual_map.get(name)
            if actual is None or not math.isfinite(actual):
                missing.append(name)
                continue
            if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                dev = abs(actual - expected) / max(abs(expected), 1.0)
                if max_dev is None or dev > max_dev[0]:
                    max_dev = (dev, name, expected, actual)

        messages: list[str] = []
        if missing:
            missing_preview = ", ".join(missing[:10])
            messages.append(
                f"{rid}: missing/non-numeric {len(missing)} expected keys (first 10: {missing_preview})"
            )
        if max_dev is not None:
            dev, name, expected, actual = max_dev
            messages.append(
                f"{rid}: max deviation {dev:.3g} at {name}: expected {expected}, got {actual}"
            )
        if messages:
            raise AssertionError("; ".join(messages))
