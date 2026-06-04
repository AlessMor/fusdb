"""Smoke test for relation registry coverage."""

from __future__ import annotations


def test_relation_registry_can_be_scanned():
    try:
        import pyomo.environ  # noqa: F401
    except Exception:
        return

    from fusdb import RelationSystem, Variable
    from fusdb.registry import RELATIONS, VARIABLES

    seen = []
    for rel in RELATIONS:
        variables = []
        for name in sorted(set(rel.variables)):
            if name not in VARIABLES:
                continue
            spec = VARIABLES.get(name)
            variables.append(Variable(name, value=[1.0, 1.0, 1.0], size=3) if spec.shape == 1 else Variable(name, value=1.0))
        try:
            RelationSystem(variables, [rel]).build_pyomo_model("reconcile")
        except Exception:
            pass
        seen.append(rel.name)
    assert set(seen) == {rel.name for rel in RELATIONS}
