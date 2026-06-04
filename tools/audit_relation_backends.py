"""Audit relation backend support across the complete relation registry.

Run from the project root with:

    python tools/audit_relation_backends.py

The script imports every registered relation and tries to construct Pyomo
constraints using dummy variables. It reports native, external, and failed
relations. It has no assertions because some registry relations are expected to
fall back to external functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusdb import RelationSystem, Variable  # noqa: E402
from fusdb.registry import RELATIONS, VARIABLES  # noqa: E402


def variables_for_relation(rel):
    out = []
    for name in sorted(set(rel.variables)):
        if name not in VARIABLES:
            continue
        spec = VARIABLES.get(name)
        if spec.shape == 1:
            out.append(Variable(name, value=[1.0, 1.0, 1.0], size=3))
        else:
            out.append(Variable(name, value=1.0))
    return out


def main() -> int:
    try:
        import pyomo.environ  # noqa: F401
    except Exception as exc:
        print(f"Pyomo unavailable: {exc}")
        return 0

    groups: dict[str, list[str]] = {"native": [], "external": [], "failed": []}
    for rel in RELATIONS:
        try:
            system = RelationSystem(variables_for_relation(rel), [rel])
            system.build_pyomo_model("reconcile")
            groups[system.relation_backends.get(rel.name, "not_used")].append(rel.name)
        except Exception as exc:
            groups["failed"].append(f"{rel.name}: {type(exc).__name__}: {exc}")

    for backend in ("native", "external", "failed"):
        print(f"\n[{backend}] {len(groups[backend])}")
        for name in sorted(groups[backend]):
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
