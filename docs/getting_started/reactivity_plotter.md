# Reactivity Plotter

This page embeds a fusion-reactivity figure rendered from the `fusdb` reactivity
relations with `fusdb.plotting.plot_reactivity`. The figure is a static, scalable
SVG generated at site-build time, with one curve per reaction (the registry's
default parametrisation for each).

<div style="width: 100%; height: 760px; border: 1px solid #e1e4e5;">
  <iframe src="../../code_docs/reactivity_plotter.html" style="width: 100%; height: 100%; border: 0;" loading="lazy"></iframe>
</div>

## Notes

- The curves are discovered from the relation registry, so new reactions or
  parametrisations appear automatically.
- The embedded HTML is generated during `mkdocs build` and `mkdocs serve` by
  `docs/scripts/build_docs_assets.py`.
- To reproduce or customise it in your own code:

  ```python
  from fusdb.plotting import plot_reactivity
  ax = plot_reactivity()          # all reactions, log-log <sigma v> vs T_i
  ax.figure.savefig("reactivity.svg")
  ```
