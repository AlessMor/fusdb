# Interactive Pages

This section provides browser-launched interactive examples for selected parts
of `fusdb` without requiring users to install the package locally.

The implementation is intentionally minimal:

- each interactive page is a normal Markdown page in the docs site;
- a companion `.ipynb` file lives next to it in the same folder;
- the page links to Binder, which installs `fusdb` from this repository and
  opens the notebook in a temporary cloud session.

This keeps the static site simple and makes interactive content opt-in on only
the pages that need it.

## Why This Pattern

- `fusdb` depends on packages such as `pandas`, `pyomo`, and `hypernetx`, so a
  pure browser-only runtime would be much more fragile.
- `pip install fusdb` stays lightweight because docs and Binder tooling are not
  part of the default package install.
- If Binder is unavailable, the docs page still works as a normal static page.

## How To Add A New Interactive Page

1. Create `your_page.md` and `your_page.ipynb` side by side in this folder.
2. Keep one editable input cell in the notebook and leave setup/result cells
   fixed.
3. Add a Binder link that points at the notebook path in this repository.
4. Add the Markdown page to the MkDocs nav.

## Current Example

- [Reactor Playground](reactor_playground.md): load a reactor, edit one input
  cell, and inspect selected solved outputs.
