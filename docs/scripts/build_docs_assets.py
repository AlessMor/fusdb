"""Generate build-time docs assets."""

from __future__ import annotations

from pathlib import Path
import sys

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(ROOT / "src")
README_PATH = ROOT / "README.md"

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from fusdb.plotting.reactivity_plotter import render_reactivity_plotter_html


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
        + "\n"
    )


with mkdocs_gen_files.open("getting_started.md", "w") as generated_page:
    generated_page.write(render_getting_started_markdown())

with mkdocs_gen_files.open("code_docs/interactive/reactivity_plotter.html", "w") as generated_page:
    generated_page.write(
        render_reactivity_plotter_html(
            x_limits=(1.0, 5.0e2),
            y_limits=(1e-30, 1e-21),
            num_points=1200,
        )
    )
