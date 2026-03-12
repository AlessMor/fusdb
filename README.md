# fusdb

`fusdb` is a lightweight fusion reactor scenario database and solver.
It stores reactor scenarios in `reactor.yaml` files and uses Python relations to infer missing quantities and check consistency.

This repository is meant as a practical research tool, not as a scientific source. Results should be checked before being
used in analysis.

## Installation

If the package has been published to PyPI:

```bash
pip install fusdb
```

From this repository:

```bash
pip install -e .
```

To build or serve the documentation locally:

```bash
pip install -e .[docs]
mkdocs serve -f docs/mkdocs.yml
```

## Where To Look Next

- Getting started: [`docs/getting_started.md`](docs/getting_started.md)
- Code documentation: [`docs/code_docs/index.md`](docs/code_docs/index.md)
- Local full docs, including the work-in-progress knowledge base: `mkdocs serve -f docs/mkdocs.yml`
- Public docs preview: `mkdocs serve -f docs/mkdocs-public.yml`
