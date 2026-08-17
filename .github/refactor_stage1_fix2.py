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
    if label == "grouped jacobian uses packed helper":
        return text
    return original_sub_once(text, pattern, replacement, label)


runner.sub_once = sub_once
runner.stage1()

path = runner.RECONCILE
lines = path.read_text().splitlines()
needle = 'for name, start, stop, offs, scales, shape, transform in group["spans"]:'
indices = [i for i, line in enumerate(lines) if line.strip() == needle]
if len(indices) != 1:
    raise RuntimeError(f"grouped Jacobian coordinate loop: expected one match, found {len(indices)}")
i = indices[0]
expected = [
    needle,
    "local = x_new[start:stop]",
    'actual = offs * np.exp(local) if transform == "log" else offs + scales * local',
    "ns[name] = actual.copy() if shape == 1 else float(actual[0])",
    "system.apply_profile_specs(ns)",
]
actual = [line.strip() for line in lines[i:i + 5]]
if actual != expected:
    raise RuntimeError(f"unexpected grouped Jacobian coordinate block: {actual!r}")
indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
lines[i:i + 5] = [
    indent + 'system.apply_packed_values(ns, x_new, group["spans"])',
    indent + "system.apply_profile_specs(ns)",
]
path.write_text("\n".join(lines) + "\n")
