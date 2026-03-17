---

status: Online

---

# Reactivity Plotter

This page embeds a standalone Bokeh app generated from `fusdb` reactivity
relations via `fusdb.plotting`. The plotter is fully static at site-build
time: all curve data is precomputed in Python, then interactivity runs in the
browser.

<div style="width: 100%; height: 760px; border: 1px solid #e1e4e5;">
  <iframe src="reactivity_plotter.html" style="width: 100%; height: 100%; border: 0;" loading="lazy"></iframe>
</div>

## Notes

- Reactions and sources can be toggled independently.
- Axis limits are applied client-side, so no live Python kernel is required.
- The embedded HTML is generated automatically during `mkdocs build` and
  `mkdocs serve`.
- Single static plots are intentionally not wrapped in a separate API anymore;
  use the underlying reactivity relations directly with your plotting library of
  choice when you need one-off figures.
