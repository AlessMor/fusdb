"""Generate build-time documentation assets.

Run by the mkdocs ``gen-files`` plugin (see ``docs/mkdocs.yml``); it writes
virtual pages into the site at build time. The page generation is decoupled from
the ``fusdb`` runtime -- it only reads the filesystem (source tree, reactor and
registry YAML, notebooks, README), so refactors of the library never break the
docs build. The actual API extraction is left to ``mkdocstrings`` via the
``::: module`` stubs emitted here.

The one exception is the embedded figure widgets, which import ``fusdb.plotting``
to render live figures (the interactive reactivity plotter via Bokeh, the
relation graph via matplotlib). That import is wrapped in a best-effort guard so
a plotting backend failure degrades to a placeholder instead of failing the
build.

Generated pages:
  * ``getting_started/index.md``               -- landing page from the repo README
  * ``getting_started/examples/*.ipynb``       -- mirrored example notebooks
    and ``code_docs/examples/*.ipynb``
  * ``code_docs/api/**``                       -- mkdocstrings stubs mirroring src/fusdb
  * ``code_docs/api/fusdb/registry/*``         -- tables for the registry YAML files
  * ``code_docs/reactors/**``                  -- tables for the reactor input YAML files
  * ``code_docs/reactivity_plotter.html``      -- reactivity figure widget
  * ``code_docs/relations_variables_graph.html`` -- relation/variable graph widget

To extend: drop files into ``examples/``, ``src/fusdb``, ``reactors/`` or
``src/fusdb/registry`` -- the example, API, reactor and registry pages are
discovered from the filesystem automatically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mkdocs_gen_files
import yaml

# --- Repository layout (read-only sources) ----------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PKG = ROOT / "src" / "fusdb"
REGISTRY = PKG / "registry"
REACTORS = ROOT / "reactors"
README = ROOT / "README.md"
EXAMPLES = ROOT / "examples"
BIBLIOGRAPHY = ROOT / "docs" / "bibliography" / "bibliography.bib"

# --- Site layout (docs-relative output paths) -------------------------------
API_ROOT = Path("code_docs/api")
REACTOR_ROOT = Path("code_docs/reactors")
GETTING_STARTED_REACTORS = Path("getting_started/reactors.md")
GETTING_STARTED_EXAMPLES = Path("getting_started/examples")
CODE_DOC_EXAMPLES = Path("code_docs/examples")


# --- Output + markdown helpers ----------------------------------------------
def _write(path: Path | str, text: str) -> None:
    """Write one virtual page into the generated site."""
    with mkdocs_gen_files.open(str(path), "w") as page:
        page.write(text)


def _set_edit(path: Path | str, source: Path) -> None:
    """Point a generated page's "edit" link at its repository source."""
    try:
        mkdocs_gen_files.set_edit_path(str(path), source.relative_to(ROOT).as_posix())
    except Exception:
        pass


def md_escape(value: object) -> str:
    """Render a scalar as inline-table-safe markdown."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def md_inline(value: object) -> str:
    """Render any YAML value as a compact single-cell representation."""
    if isinstance(value, dict):
        return "<br>".join(f"`{k}`: {md_inline(v)}" for k, v in value.items()) if value else ""
    if isinstance(value, list):
        if not value:
            return ""
        return "<br>".join(
            md_inline(item)
            if isinstance(item, (dict, list))
            else f"`{md_escape(item).strip('`')}`"
            for item in value
        )
    return md_escape(value)


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table from already-formatted cells."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return lines


def yaml_tables(document: object, *, key_header: str = "Key") -> list[str]:
    """Render an arbitrary YAML document as readable markdown table(s).

    A mapping whose values are all mappings becomes one table with a column per
    field (the union of inner keys, in first-seen order); any other mapping
    becomes a two-column key/value table. This keeps the renderer schema
    agnostic, so new registry/reactor fields appear automatically.
    """
    if isinstance(document, dict) and document:
        specs = list(document.values())
        if all(isinstance(spec, dict) for spec in specs):
            columns: list[str] = []
            for spec in specs:
                columns.extend(key for key in spec if key not in columns)
            rows = [
                [f"`{name}`", *(md_inline(spec.get(col)) for col in columns)]
                for name, spec in document.items()
            ]
            return md_table([key_header, *columns], rows)
        return md_table(
            [key_header, "Value"],
            [[f"`{key}`", md_inline(value)] for key, value in document.items()],
        )
    if isinstance(document, list) and document:
        return [f"- {md_inline([item])}" for item in document]
    return ["_None provided._"]


# --- Bibliography -----------------------------------------------------------
def build_bibliography_index() -> None:
    """Generate a plain bibliography page without footnote backrefs."""
    if not BIBLIOGRAPHY.is_file():
        return
    try:
        from pybtex.database import parse_file
        from pybtex.plugin import find_plugin
    except Exception:
        return

    data = parse_file(str(BIBLIOGRAPHY))
    style = find_plugin("pybtex.style.formatting", "plain")()
    formatted = style.format_bibliography(data)
    lines = [
        "---",
        "title: Full Bibliography",
        "---",
        "",
        "# Full Bibliography",
        "",
    ]
    for entry in formatted:
        text = entry.text.render_as("text").strip()
        lines.append(f"- {md_escape(text)}")
    lines.append("")
    _write("bibliography/index.md", "\n".join(lines))


# --- Getting Started landing page -------------------------------------------
def build_getting_started_index() -> None:
    """Generate the Getting Started landing page from the repository README."""
    if not README.is_file():
        return
    lines = README.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]  # the section page supplies its own heading
    body = "\n".join(lines).lstrip().replace("](docs/", "](")  # repo -> docs relative links
    _write("getting_started/index.md", f"# Getting Started\n\n{body}\n")


# --- Example notebooks ------------------------------------------------------
def _iter_example_notebooks() -> list[Path]:
    """Return every repository example notebook."""
    return sorted(EXAMPLES.glob("*.ipynb"))


def _notebook_title_and_summary(path: Path) -> tuple[str, str]:
    """Return a notebook title and first markdown-cell summary.

    The summary is the first markdown cell with the top-level title removed, so
    adding a new notebook automatically gives the examples index useful text.
    """
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.stem.replace("_", " ").title(), ""

    first_markdown = next(
        (
            "".join(cell.get("source", ""))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        ),
        "",
    ).strip()
    title = path.stem.replace("_", " ").title()
    body = first_markdown
    if first_markdown.startswith("#"):
        lines = first_markdown.splitlines()
        title = lines[0].lstrip("#").strip() or title
        body = "\n".join(lines[1:]).strip()
    summary = re.sub(r"\s+", " ", body).strip()
    return title, summary


def _notebook_index_markdown(notebooks: list[Path], *, target_root: Path) -> str:
    """Render an automatically discovered notebook index."""
    rows: list[list[str]] = []
    for path in notebooks:
        title, summary = _notebook_title_and_summary(path)
        rows.append([f"[{md_escape(title)}]({path.name})", md_escape(summary)])
    return "\n".join(
        [
            "---",
            "title: Notebooks",
            "---",
            "",
            "# Notebooks",
            "",
            "Notebook-based exploratory workflows are discovered from `examples/` at build time.",
            "",
            *md_table(["Notebook", "Description"], rows),
            "",
            "## Local Usage",
            "",
            "1. Install Jupyter if needed: `pip install jupyterlab`",
            "2. Start Jupyter: `python -m jupyter lab`",
            "3. Open notebooks from the `examples/` folder.",
            "",
        ]
    )


def build_notebook_pages() -> None:
    """Mirror every example notebook and generate notebook indexes."""
    notebooks = _iter_example_notebooks()
    for root in (GETTING_STARTED_EXAMPLES, CODE_DOC_EXAMPLES):
        _write(root / "notebooks.md", _notebook_index_markdown(notebooks, target_root=root))
        _write(root / ".pages", 'nav:\n  - notebooks.md\n  - "..."\n')
        for source in notebooks:
            target = root / source.name
            _write(target, source.read_text(encoding="utf-8"))
            _set_edit(target, source)


# --- API reference tree -----------------------------------------------------
def _module_name(path: Path) -> str:
    """Return the dotted import path for a module/package file under ``src/``."""
    rel = path.relative_to(PKG.parent).with_suffix("")
    parts = rel.parts[:-1] if path.name == "__init__.py" else rel.parts
    return ".".join(parts)


def _api_doc_path(path: Path) -> Path:
    """Return the docs-relative page path for a Python source file."""
    rel = path.relative_to(PKG.parent).with_suffix("")
    if path.name == "__init__.py":
        return API_ROOT.joinpath(*rel.parts[:-1], "index.md")
    return API_ROOT.joinpath(*rel.parts).with_suffix(".md")


def _registry_doc_path(yaml_path: Path) -> Path:
    """Return the docs-relative page path for a registry YAML file."""
    return API_ROOT / "fusdb" / "registry" / f"{yaml_path.stem}_registry.md"


def _api_index_markdown(
    source_dir: Path,
    child_dirs: list[Path],
    child_modules: list[Path],
    extra_pages: list[tuple[str, str]],
) -> str:
    """Render the folder landing page for one source directory."""
    import_path = ".".join(source_dir.relative_to(PKG.parent).parts)
    lines = [f"# `{source_dir.name}`", "", f"`{import_path}`", ""]
    if child_dirs or child_modules or extra_pages:
        lines += ["## Contents", ""]
        lines += [f"- [`{child.name}`]({child.name}/index.md)" for child in child_dirs]
        lines += [f"- [`{mod.stem}`]({mod.stem}.md)" for mod in child_modules]
        lines += [f"- [{title}]({href})" for title, href in extra_pages]
        lines += [""]
    if (source_dir / "__init__.py").is_file():
        lines += ["## API", "", f"::: {import_path}", ""]
    else:
        lines += ["This package has no `__init__.py`; see the module pages above.", ""]
    return "\n".join(lines)


def _registry_markdown(yaml_path: Path) -> str:
    """Render a reference page for one registry YAML file."""
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    count = len(document) if isinstance(document, (dict, list)) else 1
    lines = [
        "---",
        f"title: {yaml_path.name} Registry",
        "---",
        "",
        f"# `{yaml_path.name}`",
        "",
        f"Source: `{yaml_path.relative_to(ROOT).as_posix()}`",
        "",
        f"This file defines {count} top-level entries.",
        "",
        *yaml_tables(document, key_header="Entry"),
        "",
    ]
    return "\n".join(lines)


def build_api_reference() -> None:
    """Generate mkdocstrings stub pages mirroring the ``src/fusdb`` tree."""
    sources = sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)
    registry_yaml = sorted(REGISTRY.glob("*.yaml"))

    # Collect every directory that (eventually) contains a Python source file.
    source_dirs: set[Path] = {PKG}
    for source in sources:
        current = source.parent
        while current.is_relative_to(PKG):
            source_dirs.add(current)
            current = current.parent

    for source_dir in sorted(source_dirs):
        child_dirs = sorted(child for child in source_dirs if child.parent == source_dir)
        child_modules = sorted(
            src for src in sources if src.parent == source_dir and src.name != "__init__.py"
        )
        extra_pages = (
            [(p.name, _registry_doc_path(p).name) for p in registry_yaml]
            if source_dir == REGISTRY
            else []
        )
        index_path = _api_doc_path(source_dir / "__init__.py")
        _write(index_path, _api_index_markdown(source_dir, child_dirs, child_modules, extra_pages))
        if (source_dir / "__init__.py").is_file():
            _set_edit(index_path, source_dir / "__init__.py")
        _write(index_path.parent / ".pages", 'nav:\n  - index.md\n  - "..."\n')

    for source in sources:
        if source.name == "__init__.py":
            continue
        module = _module_name(source)
        page_path = _api_doc_path(source)
        _write(page_path, f"# `{source.stem}`\n\n`{module}`\n\n::: {module}\n")
        _set_edit(page_path, source)

    for yaml_path in registry_yaml:
        page_path = _registry_doc_path(yaml_path)
        _write(page_path, _registry_markdown(yaml_path))
        _set_edit(page_path, yaml_path)


# --- Reactor reference pages ------------------------------------------------
def _reactor_id(path: Path) -> str:
    """Return a stable identifier for one reactor YAML file."""
    if path.name == "reactor.yaml" and path.parent != REACTORS:
        return path.parent.name
    return path.stem


def _reactor_doc_path(path: Path) -> Path:
    """Return the docs-relative page path for one reactor YAML file."""
    return REACTOR_ROOT / f"{_reactor_id(path)}.md"


def _iter_reactor_yaml() -> list[Path]:
    """Return reactor YAML files (``<id>/reactor.yaml`` then ``<id>.yaml``)."""
    nested = sorted(REACTORS.glob("*/reactor.yaml"))
    flat = sorted(REACTORS.glob("*.yaml"))
    return [*nested, *flat]


def _reactor_markdown(path: Path) -> str:
    """Render a reference page for one reactor YAML file."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    document = document if isinstance(document, dict) else {}
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    reactor_id = metadata.get("id") or _reactor_id(path)

    lines = [
        "---",
        f"title: {reactor_id} Reactor YAML",
        "---",
        "",
        f"# `{reactor_id}`",
        "",
        f"Source: `{path.relative_to(ROOT).as_posix()}`",
        "",
    ]
    for heading, key, header in (
        ("Metadata", "metadata", "Field"),
        ("Tags", "tags", "Tag"),
        ("Solver Tags", "solver_tags", "Key"),
        ("Variables", "variables", "Variable"),
    ):
        lines += [f"## {heading}", "", *yaml_tables(document.get(key), key_header=header), ""]
    return "\n".join(lines)


def _reactor_index_markdown(reactor_files: list[Path]) -> str:
    """Render the reactor reference index table."""
    rows: list[list[str]] = []
    for path in reactor_files:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        document = document if isinstance(document, dict) else {}
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        variables = document.get("variables") if isinstance(document.get("variables"), dict) else {}
        tags = document.get("tags") if isinstance(document.get("tags"), list) else []
        rows.append(
            [
                f"[`{metadata.get('id') or _reactor_id(path)}`]({_reactor_doc_path(path).name})",
                md_escape(metadata.get("name")),
                md_escape(metadata.get("year")),
                md_escape(metadata.get("organization")),
                md_inline(tags),
                f"`{len(variables)}`",
            ]
        )
    lines = [
        "---",
        "title: Reactor YAML Reference",
        "---",
        "",
        "# Reactor YAML Reference",
        "",
        "Generated summaries of the reactor input YAML files.",
        "",
        *md_table(["Reactor", "Name", "Year", "Organization", "Tags", "Variables"], rows),
        "",
    ]
    return "\n".join(lines)


def _reactor_collection_markdown(reactor_files: list[Path]) -> str:
    """Render the Getting Started reactor collection from YAML files."""
    lines = [
        "---",
        "title: Reactors",
        "---",
        "",
        "# Reactors",
        "",
        "Available reactor scenarios are discovered from `reactors/` at build time.",
        "",
    ]
    for path in reactor_files:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        document = document if isinstance(document, dict) else {}
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        reactor_id = metadata.get("id") or _reactor_id(path)
        lines += [
            f"## `{reactor_id}`",
            "",
            f"Source: `{path.relative_to(ROOT).as_posix()}`",
            "",
        ]
        for heading, key, header in (
            ("Metadata", "metadata", "Field"),
            ("Tags", "tags", "Tag"),
            ("Solver Tags", "solver_tags", "Key"),
            ("Variables", "variables", "Variable"),
        ):
            lines += [f"### {heading}", "", *yaml_tables(document.get(key), key_header=header), ""]
    return "\n".join(lines)


def build_reactor_reference() -> None:
    """Generate reference tables for the reactor input YAML files."""
    reactor_files = _iter_reactor_yaml()
    if not reactor_files:
        return
    _write(GETTING_STARTED_REACTORS, _reactor_collection_markdown(reactor_files))
    _write(REACTOR_ROOT / "index.md", _reactor_index_markdown(reactor_files))
    _write(REACTOR_ROOT / ".pages", 'nav:\n  - index.md\n  - "..."\n')
    for path in reactor_files:
        _write(_reactor_doc_path(path), _reactor_markdown(path))
        _set_edit(_reactor_doc_path(path), path)


# --- Embedded figure widgets (best-effort; never fail the build) ------------
def _widget_placeholder(name: str, reason: object) -> str:
    """Return fallback HTML shown when a figure widget cannot be rendered."""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        "<body style='font-family:sans-serif;color:#666;padding:1rem'>"
        f"<p>The {name} figure could not be generated for this build.</p>"
        f"<pre style='white-space:pre-wrap'>{reason}</pre></body></html>"
    )


def build_figure_widgets() -> None:
    """Render the embedded figure widgets from ``fusdb.plotting``.

    Generates ``code_docs/reactivity_plotter.html`` and
    ``code_docs/relations_variables_graph.html`` as standalone HTML documents
    (the iframe sources used by the docs pages). Any import/render failure is
    caught and replaced by a placeholder so the docs build never aborts here.
    """
    import sys

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))  # support source checkouts as well as installs

    widgets = {
        "reactivity_plotter.html": ("reactivity", "Fusion reactivities", _render_reactivity_widget),
        "relations_variables_graph.html": (
            "relation graph",
            "Relation–variable graph",
            _render_relation_graph_widget,
        ),
    }
    for filename, (label, title, renderer) in widgets.items():
        target = f"code_docs/{filename}"
        try:
            _write(target, renderer(title))
        except Exception as exc:  # noqa: BLE001 - widgets must never break the build
            _write(target, _widget_placeholder(label, exc))


def _render_reactivity_widget(title: str) -> str:
    """Render the interactive Bokeh reactivity plotter as embeddable HTML."""
    from fusdb.plotting.reactivity import render_reactivity_app_html

    return render_reactivity_app_html(title=title, num_points=400)


def _render_relation_graph_widget(title: str) -> str:
    """Render the relation/variable graph as embeddable HTML."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fusdb.plotting import figure_to_html, plot_relation_graph

    fig, ax = plt.subplots(figsize=(16, 10))
    plot_relation_graph(ax=ax)
    ax.set_title(title)
    html = figure_to_html(fig, fmt="svg", title=title)
    plt.close(fig)
    return html


build_getting_started_index()
build_bibliography_index()
build_notebook_pages()
build_api_reference()
build_reactor_reference()
build_figure_widgets()
