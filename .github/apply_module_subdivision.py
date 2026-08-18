from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path("src/fusdb")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def move(src: str, dst: str) -> Path:
    source = Path(src)
    target = Path(dst)
    if target.exists() and not source.exists():
        return target
    if not source.exists():
        raise RuntimeError(f"Expected source file {source} does not exist")
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite existing target {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    return target


def remove_top_level(path: Path, name: str, *, kind: type[ast.AST] = ast.FunctionDef) -> None:
    text = read(path)
    tree = ast.parse(text)
    matches = [node for node in tree.body if isinstance(node, kind) and getattr(node, "name", None) == name]
    if not matches:
        return
    if len(matches) != 1:
        raise RuntimeError(f"Expected one top-level {name!r} in {path}, found {len(matches)}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    while end < len(lines) and not lines[end].strip():
        end += 1
    write(path, "".join(lines[:start] + lines[end:]))


def remove_method(path: Path, class_name: str, method_name: str) -> None:
    text = read(path)
    tree = ast.parse(text)
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if cls is None:
        raise RuntimeError(f"Class {class_name!r} not found in {path}")
    matches = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name]
    if not matches:
        return
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {class_name}.{method_name} in {path}, found {len(matches)}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    while end < len(lines) and not lines[end].strip():
        end += 1
    write(path, "".join(lines[:start] + lines[end:]))


def replace_required(path: Path, old: str, new: str, *, count: int | None = None) -> None:
    text = read(path)
    found = text.count(old)
    if found == 0:
        if new in text:
            return
        raise RuntimeError(f"Expected text not found in {path}: {old!r}")
    if count is not None and found != count:
        raise RuntimeError(f"Expected {count} occurrence(s) in {path}, found {found}: {old!r}")
    write(path, text.replace(old, new))


# ---------------------------------------------------------------------------
# Module ownership: keep the fusdb package root focused on the four main class
# modules (Variable, Relation, RelationSystem/CompilePlan, Reactor).  Shared
# execution, I/O, profiles and numerical support live in explicit packages.
# ---------------------------------------------------------------------------

move("src/fusdb/io.py", "src/fusdb/io/results.py")
move("src/fusdb/batch.py", "src/fusdb/modes/_batch.py")
move("src/fusdb/profile_sources.py", "src/fusdb/profiles/sources.py")
move("src/fusdb/profile_system.py", "src/fusdb/profiles/system.py")
move("src/fusdb/utils/profiles.py", "src/fusdb/profiles/numerics.py")
move("src/fusdb/utils/datasets.py", "src/fusdb/registry/dataset/evaluation.py")
move("src/fusdb/utils/__init__.py", "src/fusdb/numerics/__init__.py")
move("src/fusdb/relations/utils.py", "src/fusdb/relations/composition/_helpers.py")
move("src/fusdb/plotting/tables.py", "src/fusdb/io/tables.py")
move("src/fusdb/plotting/_bokeh.py", "src/fusdb/plotting/bokeh.py")
move("src/fusdb/plotting/renderers.py", "src/fusdb/plotting/matplotlib.py")

utils_dir = ROOT / "utils"
if utils_dir.exists():
    leftovers = list(utils_dir.iterdir())
    if leftovers:
        raise RuntimeError(f"utils/ still contains files: {leftovers}")
    utils_dir.rmdir()

write(
    ROOT / "profiles/__init__.py",
    '"""Profile ingestion, coordinate conversion, and numerical profile operations."""\n',
)

write(
    ROOT / "io/__init__.py",
    '''"""FusDB input/output: result persistence and display tables."""\n\nfrom __future__ import annotations\n\nfrom importlib import import_module\nfrom typing import Any\n\nfrom .results import load_result, save_result\n\n_TABLE_EXPORTS = {"SolvedColumn", "TableCell", "TableData", "render_table", "variable_table_data"}\n\n__all__ = ["load_result", "save_result", *_TABLE_EXPORTS]\n\n\ndef __getattr__(name: str) -> Any:\n    if name not in _TABLE_EXPORTS:\n        raise AttributeError(name)\n    return getattr(import_module(".tables", __name__), name)\n\n\ndef __dir__() -> list[str]:\n    return sorted({*globals(), *_TABLE_EXPORTS})\n''',
)

# Global textual import/path migrations.  Apply to code, tests, docs and
# notebooks so the repository has one vocabulary and no hidden legacy paths.
replacements = (
    ("fusdb.utils.datasets", "fusdb.registry.dataset.evaluation"),
    ("fusdb.utils.profiles", "fusdb.profiles.numerics"),
    ("fusdb.utils", "fusdb.numerics"),
    ("fusdb.profile_sources", "fusdb.profiles.sources"),
    ("fusdb.profile_system", "fusdb.profiles.system"),
    ("fusdb.batch", "fusdb.modes._batch"),
    ("fusdb.relations.utils", "fusdb.relations.composition._helpers"),
    ("fusdb.plotting.tables", "fusdb.io.tables"),
    ("fusdb.plotting._bokeh", "fusdb.plotting.bokeh"),
)
for path in Path(".").rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.suffix.lower() not in {".py", ".md", ".rst", ".ipynb", ".toml"}:
        continue
    try:
        text = read(path)
    except UnicodeDecodeError:
        continue
    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    # Relative imports of the old generic utilities become the numerical
    # support package.  Profile-specific relative imports are repaired below
    # after their modules have moved into fusdb.profiles.
    updated = re.sub(
        r"(?m)^(\s*from\s+\.+)utils(?=(?:\.|\s+import))",
        r"\1numerics",
        updated,
    )
    if updated != text:
        write(path, updated)

# Imports in modules that changed package depth.
sources = ROOT / "profiles/sources.py"
text = read(sources)
text = text.replace("from .relation import", "from ..relation import")
text = text.replace("from .registry import", "from ..registry import")
text = text.replace("from .variable import", "from ..variable import")
text = text.replace("from .numerics.profiles import", "from .numerics import")
write(sources, text)

system = ROOT / "profiles/system.py"
text = read(system)
text = text.replace("from .profile_sources import", "from .sources import")
text = text.replace("from .relation import", "from ..relation import")
text = text.replace("from .registry import", "from ..registry import")
text = text.replace("from .registry.coordinate_variables import", "from ..registry.coordinate_variables import")
text = text.replace("from .relationsystem import", "from ..relationsystem import")
text = text.replace("from .variable import", "from ..variable import")
write(system, text)

numerics = ROOT / "numerics/__init__.py"
text = read(numerics)
text = text.replace('"""General utilities for FusDB numeric relation solving."""', '"""Low-level numerical primitives shared by FusDB core modules."""')
text = text.replace("from .profiles import line_average, trapezoid, volume_average", "from ..profiles.numerics import line_average, trapezoid, volume_average")
write(numerics, text)

# Root/public imports and Reactor's shared execution/I/O imports.
root_init = ROOT / "__init__.py"
text = read(root_init)
text = text.replace("from .profile_system import build_relation_system", "from .profiles.system import build_relation_system")
text = text.replace("from .plotting.tables import SolvedColumn, render_table, variable_table_data", "from .io.tables import SolvedColumn, render_table, variable_table_data")
write(root_init, text)

reactor = ROOT / "reactor.py"
text = read(reactor)
text = text.replace("from .batch import map_chunks, parallel_chunk_size", "from .modes._batch import map_chunks, parallel_chunk_size")
text = text.replace("from .profile_system import build_relation_system", "from .profiles.system import build_relation_system")
text = text.replace("from .plotting.tables import SolvedColumn, _table_column", "from .io.tables import SolvedColumn, _table_column")
write(reactor, text)

# ---------------------------------------------------------------------------
# Plotting: tables are I/O/presentation, while backend rendering is explicitly
# split into Matplotlib and Bokeh modules.  Scientific plotting modules remain
# domain-specific data/app builders.
# ---------------------------------------------------------------------------

# Move TableCell/TableData to io.tables so core table formatting no longer
# belongs to plotting.data.
plot_data = ROOT / "plotting/data.py"
data_text = read(plot_data)
data_tree = ast.parse(data_text)
table_nodes = [node for node in data_tree.body if isinstance(node, ast.ClassDef) and node.name in {"TableCell", "TableData"}]
if table_nodes:
    lines = data_text.splitlines(keepends=True)
    for node in sorted(table_nodes, key=lambda n: n.lineno, reverse=True):
        start = node.lineno - 2 if node.lineno >= 2 and lines[node.lineno - 2].lstrip().startswith("@dataclass") else node.lineno - 1
        end = node.end_lineno or node.lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
    write(plot_data, "".join(lines).rstrip() + "\n")

io_tables = ROOT / "io/tables.py"
text = read(io_tables)
text = text.replace("from collections.abc import Iterable, Mapping", "from collections.abc import Iterable, Mapping, Sequence")
text = text.replace("from typing import Any, NamedTuple", "from dataclasses import dataclass\nfrom typing import Any, NamedTuple")
text = text.replace("from .data import TableCell, TableData\n", "")
marker = "from ..registry import VARIABLES\n"
table_defs = '''\n\n@dataclass(frozen=True)\nclass TableCell:\n    """A display-ready table cell, independent of HTML or text rendering."""\n\n    text: str\n    foreground: str = "#000000"\n    background: str = ""\n    tooltip: str = ""\n\n\n@dataclass(frozen=True)\nclass TableData:\n    """A table with already-formatted cells for HTML and plain-text renderers."""\n\n    headers: Sequence[str]\n    rows: Sequence[tuple[str, Sequence[TableCell]]]\n    header_colors: Sequence[str] = ()\n\n    def __post_init__(self) -> None:\n        headers = tuple(self.headers)\n        rows = tuple((str(name), tuple(cells)) for name, cells in self.rows)\n        if any(len(cells) != len(headers) for _, cells in rows):\n            raise ValueError("Every TableData row must contain one cell per header.")\n        colors = tuple(self.header_colors) or tuple("#000000" for _ in headers)\n        if len(colors) != len(headers):\n            raise ValueError("TableData header_colors must match headers.")\n        object.__setattr__(self, "headers", headers)\n        object.__setattr__(self, "rows", rows)\n        object.__setattr__(self, "header_colors", colors)\n'''
if "class TableCell" not in text:
    if marker not in text:
        raise RuntimeError("Could not locate io.tables registry import insertion point")
    text = text.replace(marker, marker + table_defs, 1)
write(io_tables, text)

# Extract the Bokeh CurveSet renderer from the old mixed renderers module and
# place it with the rest of the Bokeh backend.
matplotlib_path = ROOT / "plotting/matplotlib.py"
mat_text = read(matplotlib_path)
mat_tree = ast.parse(mat_text)
bokeh_node = next((node for node in mat_tree.body if isinstance(node, ast.FunctionDef) and node.name == "bokeh_curve_set"), None)
if bokeh_node is not None:
    mat_lines = mat_text.splitlines(keepends=True)
    bokeh_source = "".join(mat_lines[bokeh_node.lineno - 1 : bokeh_node.end_lineno])
    start = bokeh_node.lineno - 1
    end = bokeh_node.end_lineno or bokeh_node.lineno
    while end < len(mat_lines) and not mat_lines[end].strip():
        end += 1
    del mat_lines[start:end]
    mat_text = "".join(mat_lines)
    mat_text = mat_text.replace("from ._bokeh import move_legends_below\n", "")
    mat_text = mat_text.replace("from .bokeh import move_legends_below\n", "")
    write(matplotlib_path, mat_text)

    bokeh_path = ROOT / "plotting/bokeh.py"
    bokeh_text = read(bokeh_path)
    if "from .data import CurveSet" not in bokeh_text:
        insert_after = "import numpy as np\n"
        if insert_after not in bokeh_text:
            raise RuntimeError("Could not locate Bokeh import insertion point")
        bokeh_text = bokeh_text.replace(insert_after, insert_after + "\nfrom .data import CurveSet\n", 1)
    if "def bokeh_curve_set(" not in bokeh_text:
        bokeh_text = bokeh_text.rstrip() + "\n\n\n" + bokeh_source.rstrip() + "\n"
    write(bokeh_path, bokeh_text)

# Update plotting backend imports and lazy exports.
for filename in ("reactivity.py", "atomic_physics.py"):
    path = ROOT / "plotting" / filename
    text = read(path)
    text = text.replace("from ._bokeh import (", "from .bokeh import (")
    text = text.replace("from .renderers import bokeh_curve_set", "from .bokeh import bokeh_curve_set")
    write(path, text)

relation_graph = ROOT / "plotting/relation_graph.py"
text = read(relation_graph)
text = text.replace("from ._bokeh import move_legends_below", "from .bokeh import move_legends_below")
write(relation_graph, text)

plot_init = ROOT / "plotting/__init__.py"
text = read(plot_init)
text = text.replace('* :mod:`fusdb.plotting.renderers`  -- explicit Matplotlib/Bokeh renderers', '* :mod:`fusdb.plotting.matplotlib` -- Matplotlib renderers\n* :mod:`fusdb.plotting.bokeh`      -- Bokeh renderers and explorer scaffolding')
text = text.replace('* :mod:`fusdb.plotting.tables`     -- variable-table preparation/rendering\n', '')
text = text.replace('    "TableCell": "data",\n', '')
text = text.replace('    "TableData": "data",\n', '')
text = text.replace('    "bokeh_curve_set": "renderers",', '    "bokeh_curve_set": "bokeh",')
text = text.replace('    "plot_curve_set": "renderers",', '    "plot_curve_set": "matplotlib",')
text = text.replace('    "plot_field_map": "renderers",', '    "plot_field_map": "matplotlib",')
for old in (
    '    "variable_table_data": "tables",\n',
    '    "render_table": "tables",\n',
    '    "SolvedColumn": "tables",\n',
):
    text = text.replace(old, '')
write(plot_init, text)

# Any explicit imports of the old mixed renderer module in tests/examples can
# be routed by symbol without keeping a compatibility module.
for path in Path(".").rglob("*.py"):
    if ".git" in path.parts:
        continue
    text = read(path)
    updated = text.replace(
        "from fusdb.plotting.renderers import bokeh_curve_set",
        "from fusdb.plotting.bokeh import bokeh_curve_set",
    )
    updated = updated.replace(
        "from fusdb.plotting.renderers import plot_curve_set, plot_field_map",
        "from fusdb.plotting.matplotlib import plot_curve_set, plot_field_map",
    )
    updated = updated.replace(
        "from fusdb.plotting.renderers import plot_curve_set",
        "from fusdb.plotting.matplotlib import plot_curve_set",
    )
    updated = updated.replace(
        "from fusdb.plotting.renderers import plot_field_map",
        "from fusdb.plotting.matplotlib import plot_field_map",
    )
    if updated != text:
        write(path, updated)

# ---------------------------------------------------------------------------
# Requested deletions/inlining.
# ---------------------------------------------------------------------------

# Reactor: remove one dead forwarding wrapper and inline two single-use helpers.
reactor = ROOT / "reactor.py"
text = read(reactor)
if text.count("_with_sustainment_guards") > 1:
    raise RuntimeError("_with_sustainment_guards unexpectedly has live callers")
write(reactor, text)
remove_top_level(reactor, "_with_sustainment_guards")

text = read(reactor)
old = "_regime_warning(declared, selected, mode),"
if old in text:
    text = text.replace(
        old,
        '(\n                    f"Declared {declared} operating condition is inconsistent with confinement-mode thresholds; "\n                    f"switched to {selected} for {mode}."\n                ),',
        1,
    )
write(reactor, text)
remove_top_level(reactor, "_regime_warning")

text = read(reactor)
old = "name: var.clone(value=_copy_value(var.input_value))"
if old in text:
    text = text.replace(
        old,
        "name: var.clone(value=var.input_value.copy() if isinstance(var.input_value, np.ndarray) else var.input_value)",
        1,
    )
write(reactor, text)
remove_top_level(reactor, "_copy_value")

# Variable: the ndarray copy is a one-use implementation detail of ingestion.
variable = ROOT / "variable.py"
replace_required(
    variable,
    'object.__setattr__(self, "input_value", self._copy_value(self.value))',
    'object.__setattr__(self, "input_value", self.value.copy() if isinstance(self.value, np.ndarray) else self.value)',
)
remove_method(variable, "Variable", "_copy_value")

# Source profile construction: the direct-data builder only forwarded one call
# after worker reconstruction metadata was removed.  Keep one construction path.
sources = ROOT / "profiles/sources.py"
remove_top_level(sources, "_source_profile_relation_from_data")
text = read(sources)
tree = ast.parse(text)
node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_profile_relation"), None)
if node is None:
    raise RuntimeError("source_profile_relation not found")
lines = text.splitlines(keepends=True)
start = node.lineno - 1
end = node.end_lineno or node.lineno
replacement = '''def source_profile_relation(variable: Variable, *, average_name: str | None) -> Relation:\n    """Build the ordinary relation that maps one immutable source profile.\n\n    For a movable supplied profile, ``average_name`` is the sole amplitude\n    degree of freedom and the dynamically reinterpolated source curve supplies\n    only the shape. For a fixed supplied profile, the absolute source values\n    are mapped directly and no amplitude variable is introduced.\n    """\n    name = variable.name\n    coordinate = variable.coordinate or "rho"\n    fixed = bool(variable.fixed)\n    if not fixed and average_name is None:\n        raise ValueError(f"Movable source profile {name!r} has no registered volume-average variable.")\n\n    func = partial(\n        _evaluate_source_profile,\n        source_values=np.asarray(variable.input_value, dtype=float).copy(),\n        source_coordinate=_source_grid(variable),\n        fixed=fixed,\n    )\n    input_names: tuple[str, ...] = ()\n    argument_names: tuple[str, ...] = ()\n    if not fixed:\n        input_names += (str(average_name),)\n        argument_names += ("average",)\n    if coordinate != "rho":\n        input_names += (coordinate,)\n        argument_names += ("mapping",)\n\n    return Relation(\n        name=f"Source profile {name}" if coordinate == "rho" else f"Source profile {name} on {coordinate}",\n        func=func,\n        input_names=input_names,\n        outputs=(name,),\n        tags=("profile",),\n        constant_names=("rho", "w_V", "v_norm"),\n        dependency="generated_profile",\n        function_name=f"source_profile_{name}" if coordinate == "rho" else f"source_profile_{name}_on_{coordinate}",\n        argument_names=argument_names,\n        source_kind="source_profile",\n        source_name=name,\n    )\n'''
lines[start:end] = [replacement]
write(sources, "".join(lines))

# VariableSpec owns SciPy packing bounds, so inline its one-use wrapper and
# delete scipy_bounds from the shared numerical namespace.
variable_registry = ROOT / "registry/variable_registry.py"
text = read(variable_registry)
text = text.replace("    scipy_bounds,\n", "")
old = "        s_lo, s_hi = scipy_bounds(self.solver_domain, zero_tol=ZERO_TOL)"
new = '''        s_lo_raw, s_hi_raw = domain_bounds_for_solver(self.solver_domain, zero_tol=ZERO_TOL)\n        s_lo = -np.inf if s_lo_raw is None else float(s_lo_raw)\n        s_hi = np.inf if s_hi_raw is None else float(s_hi_raw)'''
if old not in text and new not in text:
    raise RuntimeError("VariableSpec scipy_bounds call not found")
text = text.replace(old, new)
write(variable_registry, text)
remove_top_level(numerics, "scipy_bounds")

# safe_max_abs is used only by Relation._scaled_comparison; keeping the finite
# magnitude calculation at that scale-selection site is shorter than another
# package-wide helper.
relation = ROOT / "relation.py"
text = read(relation)
text = text.replace("safe_max_abs, ", "")
old = "            base_scale = max(safe_max_abs(lhs), safe_max_abs(rhs), 1.0)"
new = '''            magnitudes = [1.0]\n            for value in (lhs, rhs):\n                try:\n                    arr = np.asarray(value, dtype=float).reshape(-1)\n                except Exception:\n                    continue\n                finite = arr[np.isfinite(arr)]\n                if finite.size:\n                    magnitudes.append(float(np.max(np.abs(finite))))\n            base_scale = max(magnitudes)'''
if old not in text and new not in text:
    raise RuntimeError("Relation safe_max_abs call not found")
text = text.replace(old, new)
write(relation, text)
remove_top_level(numerics, "safe_max_abs")

# The moved numerical package must not import a non-existent local profiles
# module after the preceding AST edits.
text = read(numerics)
text = text.replace("from .profiles import", "from ..profiles.numerics import")
write(numerics, text)

# ---------------------------------------------------------------------------
# Structural postconditions: fail the workflow before tests if the refactor left
# stale module paths or root-level support modules behind.
# ---------------------------------------------------------------------------

for forbidden in (
    ROOT / "io.py",
    ROOT / "batch.py",
    ROOT / "profile_sources.py",
    ROOT / "profile_system.py",
    ROOT / "utils",
    ROOT / "plotting/renderers.py",
    ROOT / "plotting/_bokeh.py",
    ROOT / "plotting/tables.py",
    ROOT / "relations/utils.py",
):
    if forbidden.exists():
        raise RuntimeError(f"Old module still exists: {forbidden}")

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
for path in Path("src").rglob("*.py"):
    text = read(path)
    stale = [token for token in stale_tokens if token in text]
    if stale:
        raise RuntimeError(f"Stale module reference(s) {stale} in {path}")

for token in ("safe_max_abs", "scipy_bounds", "_with_sustainment_guards", "_regime_warning"):
    matches = []
    for path in ROOT.rglob("*.py"):
        if token in read(path):
            matches.append(str(path))
    if matches:
        raise RuntimeError(f"Deleted helper {token!r} still referenced in {matches}")

# Parse every source and test module immediately; import/runtime behavior is
# covered by pytest in the workflow.
for path in [*Path("src").rglob("*.py"), *Path("tests").rglob("*.py")]:
    ast.parse(read(path), filename=str(path))

print("module subdivision and inlining refactor applied")
