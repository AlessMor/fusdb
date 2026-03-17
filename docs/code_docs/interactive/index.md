---

status: Online

---

# Interactive Pages

This section provides interactive examples for selected parts of `fusdb`
without requiring a full local setup.

Two delivery patterns are used:

- standalone HTML widgets embedded directly in the docs site;
- Jupyter notebooks rendered directly in the docs site through `mkdocs-jupyter`.

This keeps the site simple while still allowing richer worked examples when
browser-only assets are not enough.

## Why These Patterns

- Browser-only assets are ideal when the interaction can be precomputed and
  shipped as static HTML.
- Some workflows are better represented as notebooks with saved outputs and code
  cells, instead of forcing them into a custom browser-only widget.
- `pip install fusdb` stays lightweight because docs-only tooling is not part
  of the default package install.

## How To Add A New Interactive Page

1. Decide whether the interaction should be a standalone HTML widget or a rendered notebook page.
2. For standalone assets, generate `your_asset.html` and embed it from `your_page.md` with an iframe.
3. For notebook workflows, place `your_page.ipynb` in the docs tree and add the notebook itself to the MkDocs nav.
4. If the notebook should display results in docs, commit saved outputs or choose build-time execution explicitly.

## Current Examples

- [Reactivity Plotter](reactivity_plotter.md): standalone Bokeh widget embedded
  directly in the docs site.
- [Reactor Playground](reactor_playground.ipynb): load a reactor, edit one
  input cell, and inspect selected solved outputs in a rendered notebook page.
