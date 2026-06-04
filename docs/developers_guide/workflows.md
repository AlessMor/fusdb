# Workflow Playbooks

Practical workflows for extending and validating relation-driven models.

## Add a New Relation

1. Place the module in the correct physics domain under `src/fusdb_pyomo/...`.
2. Declare relations with the `@relation(outputs=..., tags=...)` decorator.
3. Add hard/soft constraints as needed (use the `constraints` argument).
4. Add tests covering forward evaluation and at least one verification/solve path.

## Add a New Variable

1. Add variable metadata in `src/fusdb_pyomo/registry/variables.yaml`.
2. Define default behavior and tolerances where required.
3. Update registry defaults if needed.
4. Ensure variable unit and dimensionality match all relations that consume it.

## Diagnose Inconsistencies

1. Run `Reactor.run()` or `RelationSystem.run(mode="verify")` to evaluate
  relations without committing solver changes.
2. Inspect `relation_status` / `variable_status` and `residuals` in the result dict.
3. Confirm variable tolerances are realistic for the physics regime.
4. If needed, constrain relation selection or order via `relation_include`,
  `relation_exclude`, or `relation_order` on the `Reactor`.

## Keep Knowledge and Code Aligned

- When formulas or assumptions change, update both:
  - relation code / tests
  - Knowledge Base pages explaining coupling assumptions
- Prefer explicit naming for profile vs volume-integrated quantities.
