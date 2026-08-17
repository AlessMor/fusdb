from __future__ import annotations

import re
from pathlib import Path

RS = Path(__file__).resolve().parents[1] / "src/fusdb/relationsystem.py"


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


rs = RS.read_text()

packed_layout = '''    def _build_packed_layout(self):
        """Analyze solver packing without mutating the installed runtime layout."""
        lower, upper, specs, uninitialized, underdetermined = [], [], [], [], []
        for name, role in sorted(self.variable_roles.items()):
            if role == "inactive" or name not in self.packed_variables:
                if role == "fixed" and self.inputs.get(name) is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue
            spec = self.variable_registry.get(name)
            size = self.profile_size if spec.shape == 1 else 1
            if spec.shape == 1 and self.inputs.get(name) is None:
                underdetermined.append(name)
            try:
                initial = [float(self.initial_value(name, index=i if spec.shape == 1 else None)) for i in range(size)]
            except Exception:
                uninitialized.append(name)
                continue
            try:
                reference = np.asarray(spec.solver_value(self.inputs[name], self.profile_size), dtype=float).reshape(-1) if self.inputs.get(name) is not None else None
            except Exception:
                reference = None
            packed = []
            for i, init in enumerate(initial):
                ref = float(reference[min(i if spec.shape == 1 else 0, reference.size - 1)]) if reference is not None and reference.size else init
                packed.append(self.pack_scalar(name, init, *spec.solver_bounds, scale_ref=ref))
            scales, offsets, lows, highs, transforms = zip(*packed)
            start = len(lower)
            lower.extend(lows)
            upper.extend(highs)
            specs.append((name, start, len(lower), np.asarray(offsets, dtype=float), np.asarray(scales, dtype=float), spec.shape, "log" if "log" in transforms else None))
        return specs, np.asarray(lower), np.asarray(upper), uninitialized, underdetermined

'''
rs = sub_once(rs, r"    def _build_packed_layout\(.*?\n    def pack\(", packed_layout + "    def pack(", "compact pure packed-layout analyzer")

packed_apply = '''    def apply_packed_values(self, values: dict[str, Any], x: np.ndarray, specs=None) -> dict[str, Any]:
        """Apply packed solver coordinates to ``values`` without completion."""
        arr = np.asarray(x, dtype=float)
        for name, start, stop, offsets, scales, shape, transform in (self.packed_specs if specs is None else specs):
            local_x = arr[start:stop]
            actual = offsets * np.exp(local_x) if transform == "log" else offsets + scales * local_x
            values[name] = actual.copy() if shape == 1 else float(actual[0])
        return values

'''
rs = sub_once(rs, r"    def apply_packed_values\(.*?\n    def unpack\(", packed_apply + "    def unpack(", "compact shared packed-coordinate transform")

RS.write_text(rs)
