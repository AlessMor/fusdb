# Relations and Variables

Relations are deterministic physics/engineering formulas expressed as Python callables with explicit inputs and one output. Variables are typed containers that store values, tolerances, provenance, and solver history. Together they let the solver infer missing values while checking consistency.

## Relation Graph

<div style="width: 100%; height: 900px; border: 1px solid #e1e4e5;">
  <iframe src="relations_variables_graph.html" style="width: 100%; height: 100%; border: 0;" loading="lazy"></iframe>
</div>

## Relations

- One relation computes exactly one output from one or more inputs.
- Relations can have hard constraints (must hold) and soft constraints (warning-only).
- Relations can also expose inverse solvers used for backward inference.

Details: [Relation Class](relation_class.md)

## Variables

- Variables are typed (`0D` scalar or `1D` profile), unit-aware containers.
- Each variable tracks current value, optional input value, tolerances, and method hints.
- Fixed variables are protected from solver overwrites.

Details: [Variable Class](variable_class.md)
