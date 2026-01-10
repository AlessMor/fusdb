import argparse
import sys
from pathlib import Path

from .loader import OPTIONAL_METADATA_FIELDS, REQUIRED_FIELDS, load_all_reactors, reactor_table


def main() -> None:
    """Entry point for the fusdb command-line interface."""
    parser = argparse.ArgumentParser(prog="fusdb", description="Fusion reactor scenario database CLI.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root (default: current directory).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available reactors found under root/ reactors.")

    show_parser = subparsers.add_parser("show", help="Show details for a single reactor by id.")
    show_parser.add_argument("reactor_id", help="Reactor id (e.g. ARC_2018)")

    args = parser.parse_args()
    root = args.root

    if args.command == "list":
        # Build a simple fixed-width table for quick CLI inspection.
        reactors = load_all_reactors(root)
        rows = reactor_table(reactors)
        columns = [
            "id",
            "reactor_family",
            "name",
            "reactor_configuration",
            "organization",
            "design_year",
            "P_fus",
        ]
        cols = list(columns)
        str_rows = [[("" if row.get(col) is None else str(row.get(col))) for col in cols] for row in rows]
        widths = [
            max(len(col), *(len(r[i]) for r in str_rows)) if str_rows else len(col)
            for i, col in enumerate(cols)
        ]

        header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))
        separator = "-+-".join("-" * widths[i] for i in range(len(cols)))
        body_lines = [" | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) for r in str_rows]
        print("\n".join([header, separator, *body_lines]))
        sys.exit(0)

    if args.command == "show":
        # Dump metadata, parameters, and artifact info for a single reactor.
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
