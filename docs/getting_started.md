# Getting Started

## Installation

1. Install the package:
   `pip install -e .`
2. Install docs tooling (MkDocs + plugins):
   `pip install -e .[docs]`

## Build Documentation

1. Run the full local docs server, including the work-in-progress Knowledge Base:
   `mkdocs serve -f docs/mkdocs.yml`
2. To preview only what will be published publicly:
   `mkdocs serve -f docs/mkdocs-public.yml`
3. Open the local URL printed in terminal (typically `http://127.0.0.1:8000`).
4. For a local public-site build:
   `mkdocs build -f docs/mkdocs-public.yml`
5. For a full local build:
   `mkdocs build -f docs/mkdocs.yml`

## Explore Reactors

The `docs/code_docs/reactor_browser.ipynb` notebook lets you compare scenarios
in `reactors/`.

1. `python -m jupyter lab` (or `python -m jupyter notebook`)
2. Open `docs/code_docs/reactor_browser.ipynb`
3. Run the cells to browse, filter, and compare reactor data
