# Source Documentation

This section contains the code-specific documentation for `fusdb`.
Use it when you need code structure, solver behavior, class interfaces,
developer workflows, or module-level APIs.

## General Overview

- [Physics Domains](physics_domains.md): how relations are grouped across the codebase.
- [Relation Interactions](relation_interactions.md): how coupled equations interact during a solve.
- [Workflow Playbooks](workflows.md): practical extension and validation tasks.
- [Relations and Variables](relations_variables.md): core concepts and graph view.
- [Profile Integration](profile_integration.md): profile-aware relation conventions.

## Class and System Reference

- [Reactor Class](reactor_class.md): class fields, methods, and examples.
- [Relation Class](relation_class.md): relation definition and solver-facing behavior.
- [Variable Class](variable_class.md): scalar/profile container behavior and tolerances.
- [RelationSystem](relationsystem_class.md): coupled solve orchestration and diagnostics.

## API Reference

- [API Overview](api/index.md): generated module reference mirroring `src/fusdb`.
- [fusdb Package Tree](api/fusdb/index.md): root package and subpackages.
- [Relations Package](api/fusdb/relations/index.md): relation domains and modules.
- [Registry Package](api/fusdb/registry/index.md): registry loaders and defaults.
- [Utilities Module](api/fusdb/utils.md): shared helper functions.

## Example Notebooks

- [Relation Graph Generator](examples/relation_graph_generator.ipynb): generate graph visualizations for relations and variables.
- [Read ENDF-6 Format](examples/read_ENDF6-format.ipynb): inspect MF=3 sections and export annotated YAML tables.

## Related Sections

- [Getting Started](../getting_started/index.md): installation, usage guides, and user-facing notebooks.
- [Knowledge Base](../knowledge_base/index.md): physics notes and shared bibliography.
