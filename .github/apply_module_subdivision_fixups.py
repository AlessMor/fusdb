from __future__ import annotations

import ast
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# Tables are now I/O/presentation data, not plotting data.  Keep the plotting
# data module limited to Curve/CurveSet/FieldMap and update the one test that
# intentionally imports the data classes directly.
path = Path("tests/test_plotting_data.py")
text = read(path)
text = text.replace(
    "from fusdb.plotting.data import Curve, CurveSet, FieldMap, TableCell, TableData\n",
    "from fusdb.plotting.data import Curve, CurveSet, FieldMap\nfrom fusdb.io.tables import TableCell, TableData\n",
)
write(path, text)

# Keep the plotting package documentation aligned with the new backend split
# and with tables no longer living under plotting.
path = Path("src/fusdb/plotting/__init__.py")
text = read(path)
text = text.replace(
    "Submodules are imported lazily (PEP 562): the matplotlib/bokeh plotters need\n"
    "the ``plotting`` extra, while :mod:`fusdb.plotting.tables` is dependency-free\n"
    "and is imported by the core package -- accessing a plotter name here must not\n"
    "drag matplotlib into every ``import fusdb``.\n",
    "Submodules are imported lazily (PEP 562): the Matplotlib/Bokeh plotters need\n"
    "the ``plotting`` extra, so importing :mod:`fusdb.plotting` does not drag either\n"
    "backend into every ``import fusdb``.\n",
)
write(path, text)

# The package root should now contain implementation files only for the main
# public abstractions.  Everything else belongs to a named subsystem package.
root = Path("src/fusdb")
allowed_root_py = {"__init__.py", "variable.py", "relation.py", "relationsystem.py", "reactor.py"}
actual_root_py = {item.name for item in root.glob("*.py")}
if actual_root_py != allowed_root_py:
    raise RuntimeError(
        f"Unexpected root-level modules after subdivision: {sorted(actual_root_py - allowed_root_py)}; "
        f"missing: {sorted(allowed_root_py - actual_root_py)}"
    )

# No old generic/support module paths should remain in Python source.
stale_tokens = (
    "fusdb.utils",
    "fusdb.profile_system",
    "fusdb.profile_sources",
    "fusdb.batch",
    "fusdb.plotting.tables",
    "fusdb.plotting.renderers",
    "fusdb.plotting._bokeh",
    "fusdb.relations.utils",
)
for source in Path("src").rglob("*.py"):
    text = read(source)
    stale = [token for token in stale_tokens if token in text]
    if stale:
        raise RuntimeError(f"Stale module reference(s) {stale} in {source}")

for source in [*Path("src").rglob("*.py"), *Path("tests").rglob("*.py")]:
    ast.parse(read(source), filename=str(source))

print("module subdivision fixups applied")
