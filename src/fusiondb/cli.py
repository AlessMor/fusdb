import argparse
import sys
from pathlib import Path
from typing import Iterable

from .loader import (
    OPTIONAL_METADATA_FIELDS,
    PARAMETER_SECTIONS,
    REQUIRED_FIELDS,
    SECTION_FIELDS,
    load_all_reactors,
    reactor_table,
)


def _format_table(rows: list[dict[str, object]], columns: Iterable[str]) -> str:
    cols = list(columns)
    str_rows = [
        ["" if row.get(col) is None else str(row.get(col)) for col in cols] for row in rows
    ]
    widths = [
        max(len(col), *(len(r[i]) for r in str_rows)) if str_rows else len(col)
        for i, col in enumerate(cols)
    ]

    header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))
    separator = "-+-".join("-" * widths[i] for i in range(len(cols)))
    body_lines = [" | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) for r in str_rows]

    table_lines = [header, separator, *body_lines]
    return "\n".join(table_lines)


def _cmd_list(root: Path) -> int:
    reactors = load_all_reactors(root)
    rows = reactor_table(reactors)
    columns = ["id", "reactor_class", "name", "reactor_configuration", "organization", "design_year", "P_fus"]
    table = _format_table(rows, columns)
    print(table)
    return 0


def _cmd_show(root: Path, reactor_id: str) -> int:
    reactors = load_all_reactors(root)
    reactor = reactors.get(reactor_id)
    if reactor is None:
        print(f"Reactor '{reactor_id}' not found under {root / 'reactors'}", file=sys.stderr)
        return 1

    print("Metadata:")
    metadata_fields = REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS
    for field in metadata_fields:
        value = getattr(reactor, field)
        label = field
        print(f"  {label}: {'' if value is None else value}")

    print("\nParameters:")
    printed_scalar = False
    for section in PARAMETER_SECTIONS:
        section_fields = SECTION_FIELDS.get(section, [])
        section_label = section.replace("_", " ")
        section_printed = False
        for field in section_fields:
            value = getattr(reactor, field)
            if value is not None:
                if not section_printed:
                    print(f"  {section_label}:")
                    section_printed = True
                printed_scalar = True
                print(f"    {field}: {value}")
    if not printed_scalar:
        print("  (none)")

    print("\nArtifacts:")
    if reactor.has_density_profile():
        print(f"  density_profile_file: {reactor.density_profile_file}")
        print(f"  density_profile_x_axis: {reactor.density_profile_x_axis}")
        print(f"  density_profile_y_dataset: {reactor.density_profile_y_dataset}")
        print(f"  density_profile_x_unit: {reactor.density_profile_x_unit}")
        print(f"  density_profile_y_unit: {reactor.density_profile_y_unit}")
    else:
        print("  density_profile: none")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="fusiondb", description="Fusion reactor scenario database CLI.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (default: current directory).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available reactors found under root/ reactors.")

    show_parser = subparsers.add_parser("show", help="Show details for a single reactor by id.")
    show_parser.add_argument("reactor_id", help="Reactor id (e.g. ARC_2018)")

    args = parser.parse_args()
    root = args.root

    if args.command == "list":
        exit_code = _cmd_list(root)
    elif args.command == "show":
        exit_code = _cmd_show(root, args.reactor_id)
    else:
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
