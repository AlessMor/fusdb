# Reactivity Plotter

This page embeds an interactive Bokeh plotter of the fusion reactivities
`⟨σv⟩(T_i)`, built from the `fusdb` reactivity relations by
`fusdb.plotting.reactivity_app`. Every reaction/parametrisation in the relation
registry is a curve; all curve data is precomputed at site-build time, then the
interactivity runs client-side with no Python kernel.

<div style="width: 100%; height: 760px; border: 1px solid #e1e4e5;">
  <iframe src="../../code_docs/reactivity_plotter.html" style="width: 100%; height: 100%; border: 0;" loading="lazy"></iframe>
</div>

## Notes

- Filter curves with the **Reactions** and **Sources** button groups, or click a
  legend entry to hide a single curve; pan/zoom with the toolbar, or set explicit
  log-axis limits and press **Apply limits**.
- The curves are discovered from the relation registry, so new reactions or
  parametrisations appear automatically.
- The embedded HTML is generated during `mkdocs build` and `mkdocs serve` by
  `docs/scripts/build_docs_assets.py`.
- To reproduce or customise it in your own code:

  ```python
  from fusdb.plotting.reactivity_app import save_reactivity_app_html
  save_reactivity_app_html("reactivity.html")   # standalone interactive page
  ```
