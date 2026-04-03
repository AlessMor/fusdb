"""Generate build-time docs assets."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import mkdocs_gen_files
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(ROOT / "src")
PYTHON_SOURCE_ROOT = ROOT / "src" / "fusdb"
REGISTRY_SOURCE_ROOT = PYTHON_SOURCE_ROOT / "registry"
REACTORS_SOURCE_ROOT = ROOT / "reactors"
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
from fusdb.registry import load_allowed_reactions, load_allowed_variables

_, _ALIAS_TO_VARIABLE, _DEFAULT_UNITS = load_allowed_variables()
_DEFAULT_METHODS: dict[str, str] = {}
for _reaction_name, _reaction_spec in load_allowed_reactions().items():
    if not isinstance(_reaction_spec, dict):
        continue
    _sigmav_variable = _reaction_spec.get("sigmav_variable")
    _default_method = _reaction_spec.get("default_method")
    if isinstance(_sigmav_variable, str) and isinstance(_default_method, str):
        _DEFAULT_METHODS[_sigmav_variable] = f"{_reaction_name} reactivity {_default_method}"


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
    extra_pages: list[tuple[str, str]],
) -> str:
    """Render one generated folder landing page for the API tree.

    Args:
        source_dir: Source directory mirrored into the docs tree.
        child_dirs: Immediate child directories exposed as subpages.
        child_modules: Immediate child Python modules excluding ``__init__.py``.
        extra_pages: Generated non-Python reference pages shown in the folder index.

    Returns:
        Markdown content for the generated landing page.
    """
    title = source_dir.name
    import_path = ".".join(source_dir.relative_to(PYTHON_SOURCE_ROOT.parent).parts)
    init_path = source_dir / "__init__.py"
    lines = [f"# `{title}`", "", f"`{import_path}`", ""]

    if child_dirs or child_modules or extra_pages:
        lines.extend(["## Contents", ""])
        for child_dir in child_dirs:
            lines.append(f"- [`{child_dir.name}`]({child_dir.name}/index.md)")
        for child_module in child_modules:
            lines.append(f"- [`{child_module.stem}`]({child_module.stem}.md)")
        for title, href in extra_pages:
            lines.append(f"- [{title}]({href})")
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


def registry_yaml_doc_path(source_path: Path) -> Path:
    """Return the generated doc path for one registry YAML file.

    Args:
        source_path: YAML file under ``src/fusdb/registry``.

    Returns:
        MkDocs-relative doc path for the generated registry page.
    """
    return Path("code_docs/api/fusdb/registry") / f"{source_path.stem}_registry.md"


def reactor_identifier_from_path(source_path: Path) -> str:
    """Return one stable reactor identifier from a YAML source path.

    Args:
        source_path: YAML file under ``reactors/``.

    Returns:
        Reactor identifier for docs page naming.
    """
    if source_path.name == "reactor.yaml" and source_path.parent != REACTORS_SOURCE_ROOT:
        return source_path.parent.name
    return source_path.stem


def reactor_yaml_doc_path(source_path: Path) -> Path:
    """Return the generated doc path for one reactor YAML file.

    Args:
        source_path: YAML file under ``reactors/``.

    Returns:
        MkDocs-relative doc path for the generated reactor page.
    """
    return Path("code_docs/reactors") / f"{reactor_identifier_from_path(source_path)}.md"


def iter_reactor_yaml_files() -> list[Path]:
    """Return supported reactor YAML file paths in deterministic order.

    Args:
        None.

    Returns:
        Sorted list of YAML files from supported reactor layouts.
    """
    nested = sorted(path for path in REACTORS_SOURCE_ROOT.glob("*/reactor.yaml") if path.is_file())
    flat = sorted(path for path in REACTORS_SOURCE_ROOT.glob("*.yaml") if path.is_file())
    return [*nested, *flat]


def escape_markdown_cell(value: object) -> str:
    """Return one value formatted for a Markdown table cell.

    Args:
        value: Scalar or simple collection to format.

    Returns:
        Markdown-safe inline text.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    text = str(value).replace("|", r"\|").replace("\n", "<br>")
    return text


def code_list(values: object) -> str:
    """Return one list of values formatted as inline code entries.

    Args:
        values: Candidate iterable of scalars.

    Returns:
        Markdown-safe inline list string.
    """
    if not isinstance(values, list) or not values:
        return ""
    return "<br>".join(f"`{escape_markdown_cell(value).strip('`')}`" for value in values)


def text_list(values: object) -> str:
    """Return one list of values formatted as plain inline text.

    Args:
        values: Candidate iterable of scalars.

    Returns:
        Markdown-safe inline list string.
    """
    if not isinstance(values, list) or not values:
        return ""
    return "<br>".join(escape_markdown_cell(value) for value in values)


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render one Markdown table from preformatted cells.

    Args:
        headers: Column labels for the table.
        rows: Row values, where each value is already Markdown-safe.

    Returns:
        Markdown lines for the complete table.
    """
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]
    for row in rows:
        normalized = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        lines.append(f"| {' | '.join(normalized)} |")
    return lines


def render_counted_summary_table(
    document: dict[str, object],
    *,
    count_label: str,
    headers: list[str],
    row_builder: Callable[[str, object], list[str] | None],
) -> list[str]:
    """Render one standard summary section with count and table.

    Args:
        document: Mapping to summarize.
        count_label: Plural noun used in the count sentence.
        headers: Table column labels.
        row_builder: Callback returning one table row per mapping item.

    Returns:
        Markdown lines for the summary section.
    """
    # Build rows from each top-level entry and skip unsupported shapes.
    rows: list[list[str]] = []
    for key, value in document.items():
        row = row_builder(key, value)
        if row is not None:
            rows.append(row)

    # Emit a consistent summary layout across registry pages.
    lines = [
        "## Summary",
        "",
        f"{len(document)} {count_label} are registered in this file.",
        "",
    ]
    lines.extend(render_markdown_table(headers, rows))
    lines.append("")
    return lines


def render_registry_yaml_summary(stem: str, document: object) -> list[str]:
    """Render the summary section for one registry YAML file.

    Args:
        stem: Base filename (without extension) for renderer selection.
        document: Parsed YAML document.

    Returns:
        Markdown lines for the registry summary.
    """
    # Route non-mapping documents through a small generic fallback.
    if not isinstance(document, dict):
        return [
            "## Summary",
            "",
            f"This file contains one top-level `{type(document).__name__}` document.",
            "",
        ]

    # Render table-heavy registries with compact shared table builders.
    if stem == "allowed_variables":
        return render_counted_summary_table(
            document,
            count_label="variables",
            headers=["Variable", "Default unit", "ndim", "Aliases", "Constraints", "Description"],
            row_builder=lambda name, spec: (
                None
                if not isinstance(spec, dict)
                else [
                    f"`{name}`",
                    escape_markdown_cell(spec.get("default_unit")),
                    escape_markdown_cell(spec.get("ndim", 0)),
                    code_list(spec.get("aliases")),
                    code_list(spec.get("constraints")),
                    text_list(spec.get("description")),
                ]
            ),
        )

    if stem == "allowed_species":
        return render_counted_summary_table(
            document,
            count_label="species",
            headers=["Species", "Full name", "Atomic symbol", "Z", "A", "Isotopic mass (u)"],
            row_builder=lambda name, spec: (
                None
                if not isinstance(spec, dict)
                else [
                    f"`{name}`",
                    escape_markdown_cell(spec.get("full_name")),
                    f"`{escape_markdown_cell(spec.get('atomic_symbol')).strip('`')}`",
                    escape_markdown_cell(spec.get("atomic_number")),
                    escape_markdown_cell(spec.get("atomic_mass")),
                    escape_markdown_cell(spec.get("isotopic_mass_u")),
                ]
            ),
        )

    if stem == "allowed_tags":
        lines = ["## Summary", ""]
        lines.extend(
            render_markdown_table(
                ["Group", "Values"],
                [
                    ["Reactor families", code_list(document.get("reactor_families"))],
                    ["Reactor configurations", code_list(document.get("reactor_configurations"))],
                    ["Confinement modes", code_list(document.get("confinement_modes"))],
                ],
            )
        )
        lines.extend(["", "## Solving Order", ""])
        solving_rows = [
            [f"`{tag}`", escape_markdown_cell(priority)]
            for tag, priority in (document.get("solving_order") or {}).items()
        ] if isinstance(document.get("solving_order"), dict) else []
        lines.extend(render_markdown_table(["Tag", "Priority"], solving_rows))
        lines.extend(["", "## Metadata Fields", ""])
        lines.extend(
            render_markdown_table(
                ["Required fields", "Optional fields"],
                [[code_list(document.get("required_metadata_fields")), code_list(document.get("optional_metadata_fields"))]],
            )
        )
        lines.append("")
        return lines

    if stem == "allowed_reactions":
        reaction_specs = {
            name: spec
            for name, spec in document.items()
            if isinstance(name, str) and name != "settings" and isinstance(spec, dict)
        }
        lines = [
            "## Summary",
            "",
            f"{len(reaction_specs)} reactions are registered in this file.",
            "",
        ]
        settings = document.get("settings")
        if isinstance(settings, dict):
            lines.extend(["## Settings", ""])
            settings_rows = []
            for key, value in settings.items():
                if isinstance(value, dict):
                    value_cell = "<br>".join(
                        f"`{nested_key}`: {escape_markdown_cell(nested_value)}"
                        for nested_key, nested_value in value.items()
                    )
                elif isinstance(value, list):
                    value_cell = code_list(value)
                else:
                    value_cell = escape_markdown_cell(value)
                settings_rows.append([f"`{key}`", value_cell])
            lines.extend(render_markdown_table(["Key", "Value"], settings_rows))
            lines.append("")

        lines.extend(["## Reactions", ""])
        lines.extend(
            render_markdown_table(
                ["Reaction", "Reactants", "Products", "sigmav variable", "Default method"],
                [
                    [
                        f"`{name}`",
                        code_list(spec.get("reactants")),
                        code_list(spec.get("products")),
                        f"`{escape_markdown_cell(spec.get('sigmav_variable')).strip('`')}`",
                        f"`{escape_markdown_cell(spec.get('default_method')).strip('`')}`",
                    ]
                    for name, spec in reaction_specs.items()
                ],
            )
        )
        lines.append("")
        return lines

    if stem == "constants":
        return render_counted_summary_table(
            document,
            count_label="constants",
            headers=["Constant", "Value"],
            row_builder=lambda name, value: [f"`{name}`", escape_markdown_cell(value)],
        )

    if stem == "solver_defaults":
        summary_rows = [
            [f"`{key}`", code_list(value) if isinstance(value, list) else escape_markdown_cell(value)]
            for key, value in document.items()
            if not isinstance(value, dict)
        ]
        lines = ["## Summary", ""]
        lines.extend(render_markdown_table(["Key", "Value"], summary_rows))
        lsq = document.get("lsq")
        if isinstance(lsq, dict):
            lines.extend(["", "## Least-Squares Settings", ""])
            lines.extend(
                render_markdown_table(
                    ["Key", "Value"],
                    [
                        [f"`{key}`", code_list(value) if isinstance(value, list) else escape_markdown_cell(value)]
                        for key, value in lsq.items()
                    ],
                )
            )
            lines.append("")
        return lines

    # Fallback summary for unrecognized registry files.
    lines = [
        "## Summary",
        "",
        f"This file contains {len(document)} top-level entries.",
        "",
    ]
    lines.extend(
        render_markdown_table(
            ["Key", "Value type"],
            [[f"`{key}`", f"`{type(value).__name__}`"] for key, value in document.items()],
        )
    )
    lines.append("")
    return lines


def summarize_yaml_value(value: object) -> str:
    """Return one YAML value rendered for compact table display.

    Args:
        value: YAML scalar, list, or mapping.

    Returns:
        Markdown-safe one-line or multi-line inline representation.
    """
    if isinstance(value, list):
        return code_list(value)
    if isinstance(value, dict):
        if not value:
            return ""
        return "<br>".join(
            f"`{escape_markdown_cell(key).strip('`')}`: {escape_markdown_cell(item)}"
            for key, item in value.items()
        )
    return escape_markdown_cell(value)


def canonical_variable_name(name: str) -> str:
    """Return one canonical variable name using the registry alias map.

    Args:
        name: Variable name as declared in reactor YAML.

    Returns:
        Canonical variable name when an alias is known, else the original name.
    """
    return _ALIAS_TO_VARIABLE.get(name, name)


def default_unit_for_variable(name: str) -> str:
    """Return one default unit for a variable, if available in the registry.

    Args:
        name: Variable name as declared in reactor YAML.

    Returns:
        Default unit string or an empty string when unavailable.
    """
    return str(_DEFAULT_UNITS.get(canonical_variable_name(name)) or "")


def default_method_for_variable(name: str) -> str:
    """Return one default method for a variable, if available in the registry.

    Args:
        name: Variable name as declared in reactor YAML.

    Returns:
        Default method string or an empty string when unavailable.
    """
    return _DEFAULT_METHODS.get(canonical_variable_name(name), "")


def render_reactor_yaml_markdown(source_path: Path) -> str:
    """Render one generated reference page for a reactor YAML file.

    Args:
        source_path: YAML file under ``reactors/``.

    Returns:
        Markdown content for the generated reactor page.
    """
    document = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    metadata = document.get("metadata") if isinstance(document, dict) else {}
    tags = document.get("tags") if isinstance(document, dict) else []
    solver_tags = document.get("solver_tags") if isinstance(document, dict) else {}
    variables = document.get("variables") if isinstance(document, dict) else {}

    metadata_dict = metadata if isinstance(metadata, dict) else {}
    tags_list = tags if isinstance(tags, list) else []
    solver_tags_dict = solver_tags if isinstance(solver_tags, dict) else {}
    variables_dict = variables if isinstance(variables, dict) else {}

    reactor_id = metadata_dict.get("id") or reactor_identifier_from_path(source_path)
    title = f"{reactor_id} Reactor YAML"
    source_relative = source_path.relative_to(ROOT).as_posix()
    lines = [
        "---",
        f"title: {title}",
        "---",
        "",
        f"# `{reactor_id}`",
        "",
        f"Source: `{source_relative}`",
        "",
        "## Metadata",
        "",
    ]

    if metadata_dict:
        metadata_rows = [[f"`{key}`", summarize_yaml_value(value)] for key, value in metadata_dict.items()]
        lines.extend(render_markdown_table(["Field", "Value"], metadata_rows))
    else:
        lines.extend(render_markdown_table(["Field", "Value"], [["_No metadata provided_", ""]]))

    lines.extend(
        [
            "",
            "## Tags",
            "",
            f"{code_list(tags_list) if tags_list else '_No tags provided_'}",
            "",
            "## Solver Tags",
            "",
        ]
    )

    if solver_tags_dict:
        solver_rows = [[f"`{key}`", summarize_yaml_value(value)] for key, value in solver_tags_dict.items()]
        lines.extend(render_markdown_table(["Key", "Value"], solver_rows))
    else:
        lines.extend(render_markdown_table(["Key", "Value"], [["_No solver tags provided_", ""]]))

    lines.extend(
        [
            "",
            "## Variables",
            "",
            f"{len(variables_dict)} variables are declared in this file.",
            "",
        ]
    )

    variable_rows: list[list[str]] = []
    for name, spec in variables_dict.items():
        fallback_unit = default_unit_for_variable(name)
        fallback_method = default_method_for_variable(name)
        value_cell = ""
        unit_cell = escape_markdown_cell(fallback_unit)
        fixed_cell = ""
        method_cell = escape_markdown_cell(fallback_method)
        notes_cell = ""
        if isinstance(spec, dict):
            explicit_fields = ("value", "unit", "fixed", "method")
            has_explicit_fields = any(field in spec for field in explicit_fields)
            if has_explicit_fields:
                value_cell = summarize_yaml_value(spec.get("value"))
                explicit_unit = spec.get("unit")
                unit_cell = (
                    escape_markdown_cell(explicit_unit)
                    if explicit_unit not in (None, "")
                    else escape_markdown_cell(fallback_unit)
                )
                fixed_cell = escape_markdown_cell(spec.get("fixed"))
                explicit_method = spec.get("method")
                method_cell = (
                    escape_markdown_cell(explicit_method)
                    if explicit_method not in (None, "")
                    else escape_markdown_cell(fallback_method)
                )
                extra_fields = {key: value for key, value in spec.items() if key not in explicit_fields}
                notes_cell = summarize_yaml_value(extra_fields)
            else:
                value_cell = summarize_yaml_value(spec)
        else:
            value_cell = summarize_yaml_value(spec)

        variable_rows.append([f"`{name}`", value_cell, unit_cell, fixed_cell, method_cell, notes_cell])

    if not variables_dict:
        variable_rows.append(["_No variables provided_", "", "", "", "", ""])
    lines.extend(render_markdown_table(["Variable", "Value", "Unit", "Fixed", "Method", "Notes"], variable_rows))

    lines.append("")
    return "\n".join(lines)


def render_reactor_index_markdown(reactor_yaml_files: list[Path]) -> str:
    """Render the generated index page for reactor YAML references.

    Args:
        reactor_yaml_files: Reactor YAML files to include.

    Returns:
        Markdown content for ``code_docs/reactors/index.md``.
    """
    lines = [
        "---",
        "title: Reactor YAML Reference",
        "---",
        "",
        "# Reactor YAML Reference",
        "",
        "These pages summarize the reactor input YAML files as tables.",
        "",
    ]

    index_rows: list[list[str]] = []
    for source_path in reactor_yaml_files:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        metadata = document.get("metadata") if isinstance(document, dict) else {}
        tags = document.get("tags") if isinstance(document, dict) else []
        variables = document.get("variables") if isinstance(document, dict) else {}
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        tags_list = tags if isinstance(tags, list) else []
        variables_dict = variables if isinstance(variables, dict) else {}

        reactor_id = metadata_dict.get("id") or reactor_identifier_from_path(source_path)
        name = metadata_dict.get("name", "")
        year = metadata_dict.get("year", "")
        organization = metadata_dict.get("organization", "")
        target_name = reactor_yaml_doc_path(source_path).name
        source_relative = source_path.relative_to(ROOT).as_posix()

        index_rows.append(
            [
                f"[`{reactor_id}`]({target_name})",
                escape_markdown_cell(name),
                escape_markdown_cell(year),
                escape_markdown_cell(organization),
                code_list(tags_list),
                f"`{len(variables_dict)}`",
                f"`{source_relative}`",
            ]
        )

    if not reactor_yaml_files:
        index_rows.append(["_No reactor files found_", "", "", "", "", "", ""])

    lines.extend(
        render_markdown_table(
            ["Reactor", "Name", "Year", "Organization", "Tags", "Variables", "Source"],
            index_rows,
        )
    )
    lines.append("")
    return "\n".join(lines)


def render_registry_yaml_markdown(source_path: Path) -> str:
    """Render one generated reference page for a registry YAML file.

    Args:
        source_path: YAML file under ``src/fusdb/registry``.

    Returns:
        Markdown content for the generated registry page.
    """
    document = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    title = f"{source_path.name} Registry"
    lines = [
        "---",
        f"title: {title}",
        "---",
        "",
        f"# `{source_path.name}`",
        "",
        f"Source: `src/fusdb/registry/{source_path.name}`",
        "",
    ]

    lines.extend(render_registry_yaml_summary(source_path.stem, document))

    lines.append("")
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
    registry_yaml_files = sorted(REGISTRY_SOURCE_ROOT.glob("*.yaml"))
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
        extra_pages: list[tuple[str, str]] = []
        if source_dir == REGISTRY_SOURCE_ROOT:
            extra_pages = [
                (source_path.name, registry_yaml_doc_path(source_path).name)
                for source_path in registry_yaml_files
            ]
        target_path = api_index_path(source_dir)
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(
                render_api_index_markdown(
                    source_dir,
                    child_dirs=child_dirs,
                    child_modules=child_modules,
                    extra_pages=extra_pages,
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

    for source_path in registry_yaml_files:
        target_path = registry_yaml_doc_path(source_path)
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(render_registry_yaml_markdown(source_path))
        try:
            mkdocs_gen_files.set_edit_path(target_path, source_path.relative_to(ROOT))
        except Exception:
            pass


def generate_reactor_reference_pages() -> None:
    """Generate table-based docs pages for reactor YAML inputs.

    Returns:
        None.
    """
    reactor_yaml_files = iter_reactor_yaml_files()
    if not reactor_yaml_files:
        return

    index_path = Path("code_docs/reactors/index.md")
    with mkdocs_gen_files.open(index_path, "w") as generated_page:
        generated_page.write(render_reactor_index_markdown(reactor_yaml_files))

    with mkdocs_gen_files.open(index_path.parent / ".pages", "w") as generated_page:
        generated_page.write("nav:\n  - index.md\n  - \"...\"\n")

    for source_path in reactor_yaml_files:
        target_path = reactor_yaml_doc_path(source_path)
        with mkdocs_gen_files.open(target_path, "w") as generated_page:
            generated_page.write(render_reactor_yaml_markdown(source_path))
        try:
            mkdocs_gen_files.set_edit_path(target_path, source_path.relative_to(ROOT))
        except Exception:
            pass


with mkdocs_gen_files.open("getting_started/index.md", "w") as generated_page:
    generated_page.write(render_getting_started_markdown())

copy_example_notebooks()
generate_api_reference_tree()
generate_reactor_reference_pages()

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
