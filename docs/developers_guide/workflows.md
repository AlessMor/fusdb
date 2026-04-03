# Workflow Playbooks

Practical workflows for extending and validating relation-driven models.

## Add a New Relation

1. Place the module in the correct physics domain under `src/fusdb/relations/...`.
2. Define relation with `@relation(output=..., tags=...)`.
3. Add hard/soft constraints as needed.
4. Register any inverse function or initial guess if backward solve is nontrivial.
5. Add/update tests to cover forward solve and at least one inverse/check path.

## Add a New Variable

1. Add variable metadata in `src/fusdb/registry/allowed_variables.yaml`.
2. Define default behavior/tolerances where required.
3. Update default relation loading if needed (`reactor_defaults.py` or registry defaults).
4. Ensure variable unit and dimensionality match all relations that consume it.

## Diagnose Inconsistencies

1. Run solve in check mode (`mode="check"`) to avoid mutation.
2. Inspect `violated_relations` and `likely_culprits`.
3. Confirm variable tolerances are realistic for the physics regime.
4. If needed, constrain solve order by domain/relation name via `solving_order`.

## Keep Knowledge and Code Aligned

- When formulas or assumptions change, update both:
  - relation code / tests
  - Knowledge Base pages explaining coupling assumptions
- Prefer explicit naming for profile vs volume-integrated quantities.
