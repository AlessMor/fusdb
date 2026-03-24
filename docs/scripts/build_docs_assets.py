"""Generate build-time docs assets."""

from __future__ import annotations

from pathlib import Path
import sys

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(ROOT / "src")
PYTHON_SOURCE_ROOT = ROOT / "src" / "fusdb"
README_PATH = ROOT / "README.md"
EXAMPLES_DIR = ROOT / "examples"
NOTEBOOK_TARGETS = {
    "reactivity_plots.ipynb": "getting_started/examples/reactivity_plots.ipynb",
    "reactor_browser.ipynb": "getting_started/examples/reactor_browser.ipynb",
    "tau_E_solver.ipynb": "getting_started/examples/tau_E_solver.ipynb",
    "relation_graph_generator.ipynb": "code_docs/examples/relation_graph_generator.ipynb",
    "read_ENDF6-format.ipynb": "code_docs/examples/read_ENDF6-format.ipynb",
}

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from fusdb.plotting.reactivity_plotter import render_reactivity_plotter_html
from fusdb.plotting.relation_graph import render_relation_graph_html


def render_getting_started_markdown() -> str:
    """Render the Getting Started section index from the repository README.

    Returns:
        Markdown for the generated `getting_started/index.md` page.
    """
    # Load the README so the docs landing page stays aligned with the repo summary.
    readme_text = README_PATH.read_text(encoding="utf-8")
    lines = readme_text.splitlines()

    # Drop the repository title because the section page provides its own heading.
    if lines and lines[0].startswith("# "):
        lines = lines[1:]

    # Normalize docs-relative links before embedding the README body.
    body = "\n".join(lines).lstrip()
    body = body.replace("](docs/", "](")

    return (
        "# Getting Started\n\n"
        + body
        + "\n\n"
        + "## Usage Guide\n\n"
        + "- [Reactors](reactors.md)\n"
        + "- [Reactivities](../code_docs/api/fusdb/relations/reactivities/index.md)\n"
        + "- [Reactivity plotter](reactivity_plotter.md)\n"
        + "- [Notebook index](examples/notebooks.md)\n"
        + "- [Reactor browser notebook](examples/reactor_browser.ipynb)\n"
        + "\n"
    )


def copy_example_notebooks() -> None:
    """Mirror repository notebooks into section-specific MkDocs paths.

    Returns:
        None.
    """
    # Skip notebook generation when the repository examples folder is absent.
    if not EXAMPLES_DIR.is_dir():
        return

    # Route each notebook into the section that exposes it in the docs nav.
    for source_path in sorted(EXAMPLES_DIR.glob("*.ipynb")):
        target_path = NOTEBOOK_TARGETS.get(source_path.name)
        if target_path is None:
            continue
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(source_path.read_text(encoding="utf-8"))
        try:
            mkdocs_gen_files.set_edit_path(target_path, source_path.relative_to(ROOT))
        except Exception:
            pass


def source_module_name(source_path: Path) -> str:
    """Return the import path for one Python source module.

    Args:
        source_path: Python file under ``src/fusdb``.

    Returns:
        Fully qualified module or package import path.
    """
    relative_module = source_path.relative_to(PYTHON_SOURCE_ROOT.parent).with_suffix("")
    if source_path.name == "__init__.py":
        return ".".join(relative_module.parts[:-1])
    return ".".join(relative_module.parts)


def api_doc_path(source_path: Path) -> Path:
    """Return the generated API doc path for one Python source module.

    Args:
        source_path: Python file under ``src/fusdb``.

    Returns:
        MkDocs-relative doc path for the generated page.
    """
    relative_module = source_path.relative_to(PYTHON_SOURCE_ROOT.parent).with_suffix("")
    if source_path.name == "__init__.py":
        return Path("code_docs/api").joinpath(*relative_module.parts[:-1], "index.md")
    return Path("code_docs/api").joinpath(*relative_module.parts).with_suffix(".md")


def api_index_path(source_dir: Path) -> Path:
    """Return the generated API index path for one source directory.

    Args:
        source_dir: Directory under ``src/fusdb``.

    Returns:
        MkDocs-relative folder index path.
    """
    return Path("code_docs/api").joinpath(*source_dir.relative_to(PYTHON_SOURCE_ROOT.parent).parts, "index.md")


def render_api_index_markdown(
    source_dir: Path,
    *,
    child_dirs: list[Path],
    child_modules: list[Path],
) -> str:
    """Render one generated folder landing page for the API tree.

    Args:
        source_dir: Source directory mirrored into the docs tree.
        child_dirs: Immediate child directories exposed as subpages.
        child_modules: Immediate child Python modules excluding ``__init__.py``.

    Returns:
        Markdown content for the generated landing page.
    """
    title = source_dir.name
    import_path = ".".join(source_dir.relative_to(PYTHON_SOURCE_ROOT.parent).parts)
    init_path = source_dir / "__init__.py"
    lines = [f"# `{title}`", "", f"`{import_path}`", ""]

    if child_dirs or child_modules:
        lines.extend(["## Contents", ""])
        for child_dir in child_dirs:
            lines.append(f"- [`{child_dir.name}`]({child_dir.name}/index.md)")
        for child_module in child_modules:
            lines.append(f"- [`{child_module.stem}`]({child_module.stem}.md)")
        lines.append("")

    if init_path.exists():
        lines.extend(["## API", "", f"::: {import_path}", ""])
    else:
        lines.extend(
            [
                "This source subtree has no package-level `__init__.py`; use the child module pages for the API details.",
                "",
            ]
        )

    return "\n".join(lines)


def render_api_module_markdown(source_path: Path) -> str:
    """Render one generated page for a concrete Python module.

    Args:
        source_path: Python file under ``src/fusdb``.

    Returns:
        Markdown content for the generated module page.
    """
    module_name = source_module_name(source_path)
    return "\n".join(
        [
            f"# `{source_path.stem}`",
            "",
            f"`{module_name}`",
            "",
            f"::: {module_name}",
            "",
        ]
    )


def generate_api_reference_tree() -> None:
    """Generate API pages that mirror the ``src/fusdb`` source tree.

    Returns:
        None.
    """
    source_files = sorted(
        path for path in PYTHON_SOURCE_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    )
    source_dirs: set[Path] = set()
    for source_file in source_files:
        current_dir = source_file.parent
        while current_dir.is_relative_to(PYTHON_SOURCE_ROOT):
            source_dirs.add(current_dir)
            if current_dir == PYTHON_SOURCE_ROOT:
                break
            current_dir = current_dir.parent

    for source_dir in sorted(source_dirs):
        child_dirs = sorted(child_dir for child_dir in source_dirs if child_dir.parent == source_dir)
        child_modules = sorted(
            source_file
            for source_file in source_files
            if source_file.parent == source_dir and source_file.name != "__init__.py"
        )
        target_path = api_index_path(source_dir)
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(
                render_api_index_markdown(
                    source_dir,
                    child_dirs=child_dirs,
                    child_modules=child_modules,
                )
            )

        init_path = source_dir / "__init__.py"
        if init_path.exists():
            try:
                mkdocs_gen_files.set_edit_path(target_path, init_path.relative_to(ROOT))
            except Exception:
                pass

        with mkdocs_gen_files.open(target_path.parent / ".pages", "w") as generated_page:
            generated_page.write("nav:\n  - index.md\n  - \"...\"\n")

    for source_path in source_files:
        if source_path.name == "__init__.py":
            continue
        target_path = api_doc_path(source_path)
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(render_api_module_markdown(source_path))
        try:
            mkdocs_gen_files.set_edit_path(target_path, source_path.relative_to(ROOT))
        except Exception:
            pass


with mkdocs_gen_files.open("getting_started/index.md", "w") as generated_page:
    generated_page.write(render_getting_started_markdown())

copy_example_notebooks()
generate_api_reference_tree()

with mkdocs_gen_files.open("code_docs/reactivity_plotter.html", "w") as generated_page:
    generated_page.write(
        render_reactivity_plotter_html(
            x_limits=(1.0, 5.0e2),
            y_limits=(1e-30, 1e-21),
            num_points=1200,
        )
    )

with mkdocs_gen_files.open("code_docs/relations_variables_graph.html", "w") as generated_page:
    generated_page.write(render_relation_graph_html())
