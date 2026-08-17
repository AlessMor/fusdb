from __future__ import annotations

import re
from pathlib import Path

RS = Path(__file__).resolve().parents[1] / "src/fusdb/relationsystem.py"


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


text = RS.read_text()

certify = '''    def certify_relations(self, values: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], np.ndarray, list[str], list[str]]:
        """Evaluate every active relation for certification."""
        status: dict[str, dict[str, Any]] = {}
        blocks: list[np.ndarray] = []
        errors: list[str] = []
        warnings: list[str] = []
        for rel in self.relations:
            structural = not self._relation_is_residual_relation(rel)
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            residual = np.empty(0, dtype=float)
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
            else:
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
            if structural:
                rel_status["source"] = "derived_provider"
            status[rel.name] = rel_status
            warnings.extend(rel_status.get("warnings", []))
            rel_errors = rel_status.get("errors") or ()
            if rel.enforce and rel_errors:
                prefix = "" if structural and missing else f"{rel.name}: "
                errors.extend(prefix + err for err in rel_errors)
            if structural or missing:
                continue
            if rel.enforce:
                blocks.append(residual)
            elif not rel_status["verified"]:
                if is_default_relation(rel):
                    warnings.append(f"{rel.name}: weak default not satisfied after reconciliation")
                else:
                    errors.append(f"{rel.name}: check-only applicability failed")
        residuals = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        return status, residuals, errors, warnings

'''
text = sub_once(
    text,
    r"    def certify_relations\(.*?\n    def _sparsity_dependency_graph\(",
    certify + "    def _sparsity_dependency_graph(",
    "unify enforced certification error handling",
)

failed = "        failed = set(self.uninitialized_free_variables)\n"
condition = "            if role == \"inactive\" or name not in self.packed_variables or name in failed:\n"
if text.count(failed) != 1 or text.count(condition) != 1:
    raise RuntimeError("expected one pack failure alias")
text = text.replace(failed, "", 1).replace(
    condition,
    "            if role == \"inactive\" or name not in self.packed_variables or name in self.uninitialized_free_variables:\n",
    1,
)
size = "            size = self.profile_size if spec.shape == 1 else 1\n            initial = [float(self.initial_value(name, index=i if spec.shape == 1 else None)) for i in range(size)]\n"
replacement = "            initial = [\n                float(self.initial_value(name, index=i if spec.shape == 1 else None))\n                for i in range(self.profile_size if spec.shape == 1 else 1)\n            ]\n"
if text.count(size) != 1:
    raise RuntimeError("expected one single-use pack size alias")
text = text.replace(size, replacement, 1)

RS.write_text(text)
