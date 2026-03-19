"""Generate build-time docs assets."""

from __future__ import annotations

from pathlib import Path
import sys

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(ROOT / "src")
README_PATH = ROOT / "README.md"
EXAMPLES_DIR = ROOT / "examples"

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from fusdb.plotting.reactivity_plotter import render_reactivity_plotter_html
from fusdb.plotting.relation_graph import render_relation_graph_html


def render_getting_started_markdown() -> str:
    readme_text = README_PATH.read_text(encoding="utf-8")
    lines = readme_text.splitlines()

    if lines and lines[0].startswith("# "):
        lines = lines[1:]

    body = "\n".join(lines).lstrip()
    body = body.replace("](docs/", "](")

    return (
        "---\n"
        "status: Online\n"
        "---\n\n"
        "# Getting Started\n\n"
        + body
        + "\n\n"
        + "## Usage Guide\n\n"
        + "- [Reactors](code_docs/reactors.md)\n"
        + "- [Reactivities](code_docs/reactivities.md)\n"
        + "- [Reactivity plotter](code_docs/reactivity_plotter.md)\n"
        + "- [Reactor browser notebook](examples/reactor_browser.ipynb)\n"
        + "\n"
    )


def copy_example_notebooks() -> None:
    """Mirror repository notebooks into the MkDocs file tree."""
    if not EXAMPLES_DIR.is_dir():
        return

    for source_path in sorted(EXAMPLES_DIR.glob("*.ipynb")):
        target_path = f"examples/{source_path.name}"
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(source_path.read_text(encoding="utf-8"))
        try:
            mkdocs_gen_files.set_edit_path(target_path, source_path.relative_to(ROOT))
        except Exception:
            pass


with mkdocs_gen_files.open("getting_started.md", "w") as generated_page:
    generated_page.write(render_getting_started_markdown())

copy_example_notebooks()

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
