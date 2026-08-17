from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/fusdb/relationsystem.py"
text = path.read_text()
replacements = {
    "        uninitialized: list[str] = []\n        underdetermined: list[str] = []\n": "        uninitialized, underdetermined = [], []\n",
    "            if spec.shape == 1 and self.inputs.get(name) is None:\n                underdetermined.append(name)\n": "            if spec.shape == 1 and self.inputs.get(name) is None: underdetermined.append(name)\n",
    "        lower: list[float] = []\n        upper: list[float] = []\n        specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []\n": "        lower, upper, specs = [], [], []\n",
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"expected one match for compacting block, found {text.count(old)}")
    text = text.replace(old, new, 1)
path.write_text(text)
