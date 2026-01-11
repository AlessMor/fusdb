import argparse
import sys
from pathlib import Path

from .loader import OPTIONAL_METADATA_FIELDS, REQUIRED_FIELDS, load_all_reactors


def _print_reactor_columns(reactors: dict[str, object]) -> None:
    if not reactors:
        print("(no reactors)")
        return

    reactor_ids: list[str] = []
    columns: list[list[str]] = []
    max_lines = 0
    for rid, reactor in sorted(reactors.items(), key=lambda item: item[0]):
        lines = repr(reactor).splitlines()
        reactor_ids.append(rid)
        columns.append(lines)
        max_lines = max(max_lines, len(lines))

    widths = []
    for idx, lines in enumerate(columns):
        width = max(len(reactor_ids[idx]), *(len(line) for line in lines)) if lines else len(reactor_ids[idx])
        widths.append(width)

    header = " | ".join(reactor_ids[idx].ljust(widths[idx]) for idx in range(len(reactor_ids)))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(reactor_ids)))
    body = []
    for row in range(max_lines):
        row_cells = []
        for col, lines in enumerate(columns):
            value = lines[row] if row < len(lines) else ""
            row_cells.append(value.ljust(widths[col]))
        body.append(" | ".join(row_cells))

    print("\n".join([header, separator, *body]))


def main() -> None:
    """Entry point for the fusdb command-line interface."""
    parser = argparse.ArgumentParser(prog="fusdb", description="Fusion reactor scenario database CLI.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (default: current directory).")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List available reactors found under root/ reactors.")

    show_parser = subparsers.add_parser("show", help="Show details for a single reactor by id.")
    show_parser.add_argument("reactor_id", help="Reactor id (e.g. ARC_2018)")

    args = parser.parse_args()
    root = args.root

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "list":
        # Print one column per reactor using Reactor.__repr__ output.
        reactors = load_all_reactors(root)
        _print_reactor_columns(reactors)
        sys.exit(0)

    if args.command == "show":
        # Dump metadata and parameters for a single reactor.
        reactors = load_all_reactors(root)
        reactor = reactors.get(args.reactor_id)
        if reactor is None:
            print(f"Reactor '{args.reactor_id}' not found under {root / 'reactors'}", file=sys.stderr)
            sys.exit(1)

        print("Metadata:")
        metadata_fields = REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS
        for field in metadata_fields:
            value = getattr(reactor, field)
            print(f"  {field}: {'' if value is None else value}")

        print("\nParameters:")
        if reactor.parameters:
            for key in sorted(reactor.parameters.keys()):
                value = reactor.parameters.get(key)
                tol = reactor.parameter_tolerances.get(key)
                if tol is not None:
                    print(f"  {key}: {value} (tol={tol})")
                else:
                    print(f"  {key}: {value}")
        else:
            print("  (none)")

        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
