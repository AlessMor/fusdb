# fusdb Documentation

`fusdb` is a lightweight fusion reactor scenario database and solver for fusion
studies. It combines structured reactor inputs (`reactor.yaml`) with physics
relations that infer missing quantities and check consistency.

This site is organised around three entry points:

## Getting Started

Use this path when you want to install `fusdb`, load a reactor, and start using
the main tools.

- [Installation and usage guide](getting_started.md)
- [Reactors](code_docs/reactors.md)
- [Reactivity plotter](code_docs/reactivity_plotter.md)
- [Reactor browser notebook](code_docs/reactor_browser.ipynb)

## Knowledge Base

Use this path for physics notes, definitions, and model context.

- [Knowledge Base](knowledgebase/index.md)
- [Cross Sections and Reactivities](knowledgebase/Plasma%20Physics/cross_sections_reactivities.md)
- [Physics Domains](physics_domains.md)
- [Relation Interactions](relation_interactions.md)
- [Workflow Playbooks](workflows.md)

Draft pages can stay under `docs/knowledgebase/` with `status: Draft` and are
hidden from the built site until promoted.

## Source Documentation

Use this path for module, class, and function-level reference material generated
from the source code.

- [API overview](code_docs/api/index.md)
- [Core modules](code_docs/api/core.md)
- [Relation modules](code_docs/api/relations.md)
- [Registry and utilities](code_docs/api/registry_and_utils.md)

## Local Preview

- `mkdocs serve -f docs/mkdocs.yml`
