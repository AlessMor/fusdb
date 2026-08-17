from __future__ import annotations

import importlib.util
from pathlib import Path

runner_path = Path(__file__).with_name("refactor_relationsystem.py")
spec = importlib.util.spec_from_file_location("refactor_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

original_sub_once = runner.sub_once


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    if label != "grouped jacobian uses packed helper":
        return original_sub_once(text, pattern, replacement, label)
    old = '''                for name, start, stop, offs, scales, shape, transform in group["spans"]:\n                    local = x_new[start:stop]\n                    actual = offs * np.exp(local) if transform == "log" else offs + scales * local\n                    ns[name] = actual.copy() if shape == 1 else float(actual[0])\n                system.apply_profile_specs(ns)'''
    new = '''                system.apply_packed_values(ns, x_new, group["spans"])\n                system.apply_profile_specs(ns)'''
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


runner.sub_once = sub_once
runner.stage1()
