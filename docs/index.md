# fusdb Documentation

`fusdb` is a lightweight fusion reactor scenario database and solver.
It combines structured reactor data (`reactor.yaml`) with physics relations
that infer missing quantities and check consistency.

This documentation is split into three complementary parts:

- **Main Pages**: top-level entrypoints for getting started with the code and navigation.
- **Code Docs**: user guides, class references, notebooks, and API docs for the codebase.
- **Knowledge Base**: an attempt at making a comprehensive and organized knowledge base on all that is fusion. Currently a work in progress...

## Start Here

- [Getting Started](getting_started.md)
- [Code Docs Overview](code_docs/index.md)

## Code Docs

- [Overview](code_docs/index.md)
- [Reactors](code_docs/reactors.md)
- [Relations and Variables](code_docs/relations_variables.md)
- [RelationSystem](code_docs/relationsystem_class.md)
- [API Overview](code_docs/api/index.md)
- [Notebooks](code_docs/notebooks.md)

## Knowledge Base

The Knowledge Base is currently work in progress and is excluded from the
public build.

Use the full local docs config to view it:

- `mkdocs serve -f docs/mkdocs.yml`

Use the public config to preview exactly what GitHub Pages will publish:

- `mkdocs serve -f docs/mkdocs-public.yml`
