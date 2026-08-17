from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "src/fusdb/relationsystem.py"
TEST = ROOT / "tests/test_relationsystem_lifecycle.py"


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


rs = RS.read_text()
packing = '''    def _packing_issues(self) -> tuple[list[str], list[str]]:
        """Return compile-time packing failures without installing runtime state."""
        uninitialized: list[str] = []
        underdetermined: list[str] = []
        for name, role in sorted(self.variable_roles.items()):
            if role == "inactive" or name not in self.packed_variables:
                if role == "fixed" and self.inputs.get(name) is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue
            spec = self.variable_registry.get(name)
            if spec.shape == 1 and self.inputs.get(name) is None:
                underdetermined.append(name)
            try:
                for i in range(self.profile_size if spec.shape == 1 else 1):
                    self.initial_value(name, index=i if spec.shape == 1 else None)
            except Exception:
                uninitialized.append(name)
        return uninitialized, underdetermined

    def pack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Install the numeric solver layout for the already-compiled system."""
        lower: list[float] = []
        upper: list[float] = []
        specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []
        self.uninitialized_free_variables, self.underdetermined_profiles = self._packing_issues()
        failed = set(self.uninitialized_free_variables)
        for name, role in sorted(self.variable_roles.items()):
            if role == "inactive" or name not in self.packed_variables or name in failed:
                continue
            spec = self.variable_registry.get(name)
            size = self.profile_size if spec.shape == 1 else 1
            initial = [float(self.initial_value(name, index=i if spec.shape == 1 else None)) for i in range(size)]
            try:
                reference = np.asarray(spec.solver_value(self.inputs[name], self.profile_size), dtype=float).reshape(-1) if self.inputs.get(name) is not None else None
            except Exception:
                reference = None
            start = len(lower)
            packed = [
                self.pack_scalar(
                    name,
                    init,
                    *spec.solver_bounds,
                    scale_ref=float(reference[min(i if spec.shape == 1 else 0, reference.size - 1)]) if reference is not None and reference.size else init,
                )
                for i, init in enumerate(initial)
            ]
            scales, offsets, lows, highs, transforms = zip(*packed)
            lower.extend(lows)
            upper.extend(highs)
            specs.append((name, start, len(lower), np.asarray(offsets), np.asarray(scales), spec.shape, "log" if "log" in transforms else None))
        self.packed_specs = specs
        self.packed_dim = len(lower)
        self._classify_avg_to_profile()
        self._compiler_report_cache = None
        self._packed_base_values = self.input_values()
        self._movement_plan = self._build_movement_plan()
        return np.zeros(self.packed_dim), np.asarray(lower), np.asarray(upper)

'''
rs = sub_once(
    rs,
    r"    def _build_packed_layout\(.*?\n    def required_uninitialized_free_variables\(",
    packing + "    def required_uninitialized_free_variables(",
    "replace duplicated packed-layout analysis",
)
rs = rs.replace("_specs, _lower, _upper, uninitialized, _under = self._build_packed_layout()", "uninitialized, _under = self._packing_issues()")
rs = rs.replace("_specs, _lower, _upper, uninitialized, underdetermined = self._build_packed_layout()", "uninitialized, underdetermined = self._packing_issues()")
rs = rs.replace(
    "        self.uninitialized_free_variables = list(uninitialized)\n        self.underdetermined_profiles = list(underdetermined)\n\n        # Packing detects raw profile cores because only the packed layout\n",
    "        # Packing detects raw profile cores because only the packed layout\n",
)
RS.write_text(rs)
if TEST.exists():
    TEST.write_text(TEST.read_text().replace("system._build_packed_layout()", "system._packing_issues()"))
