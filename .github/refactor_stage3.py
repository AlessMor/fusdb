from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "src/fusdb/relationsystem.py"


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


rs = RS.read_text()

packed_layout = '''    def _build_packed_layout(
        self,
    ) -> tuple[
        list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]],
        np.ndarray,
        np.ndarray,
        list[str],
        list[str],
    ]:
        """Analyze solver packing without mutating the installed runtime layout."""
        lower: list[float] = []
        upper: list[float] = []
        specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []
        uninitialized: list[str] = []
        underdetermined: list[str] = []
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
            reference = None
            if self.inputs.get(name) is not None:
                try:
                    reference = np.asarray(spec.solver_value(self.inputs[name], self.profile_size), dtype=float).reshape(-1)
                except Exception:
                    pass
            start = len(lower)
            packed = [
                self.pack_scalar(
                    name,
                    init,
                    *spec.solver_bounds,
                    scale_ref=(
                        float(reference[min(i if spec.shape == 1 else 0, reference.size - 1)])
                        if reference is not None and reference.size
                        else init
                    ),
                )
                for i, init in enumerate(initial)
            ]
            scales, offsets, lows, highs, transforms = zip(*packed)
            lower.extend(lows)
            upper.extend(highs)
            specs.append(
                (
                    name,
                    start,
                    len(lower),
                    np.asarray(offsets, dtype=float),
                    np.asarray(scales, dtype=float),
                    spec.shape,
                    "log" if "log" in transforms else None,
                )
            )
        return specs, np.asarray(lower), np.asarray(upper), uninitialized, underdetermined

'''
rs = sub_once(
    rs,
    r"    def _build_packed_layout\(.*?\n    def pack\(",
    packed_layout + "    def pack(",
    "simplify pure packed-layout builder",
)

certify = '''    def certify_relations(self, values: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], np.ndarray, list[str], list[str]]:
        """Evaluate every active relation for certification.

        Structural providers are verified on the completed namespace but stay
        outside the certificate residual vector.  Residual relations share the
        same status path and contribute rows only when enforced.
        """
        status: dict[str, dict[str, Any]] = {}
        blocks: list[np.ndarray] = []
        errors: list[str] = []
        warnings: list[str] = []
        for rel in self.relations:
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            structural = not self._relation_is_residual_relation(rel)
            if missing:
                message = f"Relation {rel.name!r} missing variables {missing}."
                rel_status = {
                    "relation": rel.name,
                    "verified": False,
                    "missing": missing,
                    "errors": [message],
                    "warnings": [],
                    "enforced": rel.enforce,
                }
                if structural:
                    rel_status["source"] = "derived_provider"
                    if rel.enforce:
                        errors.append(message)
                status[rel.name] = rel_status
                continue
            try:
                rel_status, residual = self.relation_status_and_residual(rel, values)
            except Exception as exc:
                if not structural:
                    raise
                rel_status = {
                    "relation": rel.name,
                    "verified": False,
                    "errors": [str(exc)],
                    "warnings": [],
                    "enforced": rel.enforce,
                }
                residual = np.empty(0, dtype=float)
            if structural:
                rel_status["source"] = "derived_provider"
            status[rel.name] = rel_status
            warnings.extend(rel_status.get("warnings", []))
            if structural:
                if rel.enforce and rel_status.get("errors"):
                    errors.extend(f"{rel.name}: {err}" for err in rel_status["errors"])
                continue
            if rel.enforce:
                blocks.append(residual)
                if rel_status.get("errors"):
                    errors.extend(f"{rel.name}: {err}" for err in rel_status["errors"])
            elif not rel_status["verified"]:
                if is_default_relation(rel):
                    warnings.append(f"{rel.name}: weak default not satisfied after reconciliation")
                else:
                    errors.append(f"{rel.name}: check-only applicability failed")
        residuals = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        return status, residuals, errors, warnings

'''
rs = sub_once(
    rs,
    r"    def certify_relations\(.*?\n    def _sparsity_dependency_graph\(",
    certify + "    def _sparsity_dependency_graph(",
    "unify relation certification status path",
)

RS.write_text(rs)
