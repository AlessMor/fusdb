---

status: Online

---

# Notebooks

Notebook-based exploratory workflows live under `examples/` in the repository
root and are rendered in the docs site through `mkdocs-jupyter`.

## Available Notebooks

- [Reactor Browser](../examples/reactor_browser.ipynb): compare reactors and inspect key variables.
- [Relation Graph Generator](../examples/relation_graph_generator.ipynb): build graph visualizations for relations and variables.
- [Tau_E Solver](../examples/tau_E_solver.ipynb): experiments around confinement and power-loss coupling.
- [Reactivity Plots](../examples/reactivity_plots.ipynb): open the interactive reactivity plotter from a notebook.
- [Read ENDF-6 Format](../examples/read_ENDF6-format.ipynb): inspect MF=3 sections and export annotated YAML tables.

## Local Usage

1. Install Jupyter if needed: `pip install jupyterlab`
2. Start Jupyter: `python -m jupyter lab`
3. Open notebooks from the `examples/` folder.

For website-friendly interactive examples, see the
[Reactivity Plotter](reactivity_plotter.md) and the rendered notebooks in this
section.
