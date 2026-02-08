# Relations and Variables

Relations are deterministic physics/engineering formulas expressed as Python callables with explicit inputs and one output. Variables are typed containers that store values, tolerances, provenance, and solver history. Together they let the solver infer missing values while checking consistency.

## Relation Graph

```{raw} html
<div style="width: 100%; height: 900px; border: 1px solid #e1e4e5;">
  <iframe src="relations_variables_graph.html" style="width: 100%; height: 100%; border: 0;" loading="lazy"></iframe>
</div>
```

## Relations

```{include} relation_class.md
:heading-offset: 2
```

## Variables

```{include} variable_class.md
:heading-offset: 2
```
