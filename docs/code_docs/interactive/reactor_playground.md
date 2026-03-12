# Reactor Playground

This page pairs a static explanation with a companion notebook that runs in
Binder. The notebook imports `fusdb`, loads a reactor scenario from the
repository, solves it, and prints a few selected outputs.

[Launch in Binder](https://mybinder.org/v2/gh/AlessMor/fusdb/main?urlpath=lab/tree/docs/code_docs/interactive/reactor_playground.ipynb)

[Open notebook file](reactor_playground.ipynb)

## What Users Edit

The notebook is structured so users normally edit only one input cell:

```python
reactor_id = "ARC_2015"
output_names = ["P_fus", "Q", "tau_E"]
```

Everything else can stay fixed.

## Graceful Degradation

- If Binder is unavailable, this page still documents the workflow.
- The companion notebook remains available for download or local execution.
- The public site stays a normal static MkDocs site.
