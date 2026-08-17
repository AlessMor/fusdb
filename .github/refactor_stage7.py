from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/fusdb/relationsystem.py"
text = path.read_text()
inline = "            if spec.shape == 1 and self.inputs.get(name) is None: underdetermined.append(name)\n"
expanded = "            if spec.shape == 1 and self.inputs.get(name) is None:\n                underdetermined.append(name)\n"
failed = "        failed = set(self.uninitialized_free_variables)\n"
condition = "            if role == \"inactive\" or name not in self.packed_variables or name in failed:\n"
replacement = "            if role == \"inactive\" or name not in self.packed_variables or name in self.uninitialized_free_variables:\n"
for old in (inline, failed, condition):
    if text.count(old) != 1:
        raise RuntimeError(f"expected one structural packing match, found {text.count(old)}")
text = text.replace(inline, expanded, 1).replace(failed, "", 1).replace(condition, replacement, 1)
path.write_text(text)
