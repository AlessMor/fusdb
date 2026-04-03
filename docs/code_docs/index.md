# Source Documentation

This section contains the code documentation for `fusdb`.

The main objects used in `fusdb` are:
- [Reactor](reactor_class.md)
- [Variable](variable_class.md)
- [Relation](relation_class.md)
- [RelationSystem](relationsystem_class.md)

## Reactors

A [Reactor](reactor_class.md) is an object containing all the information pertaining to a specific fusion reactor configuration. It stores metadata such as its name, configuration, year and DOI of publication of its design, solver-specific settings, and available data taken from the cited reference.

Reactor can be defined directly as an object or as yaml files. Reactor YAML inputs are summarized in the [Reactor YAML Reference](reactors/index.md). The full reactor library is available in the [`reactors/` folder](https://github.com/AlessMor/fusdb/tree/main/reactors).

## Relations, Variables and RelationSystem

Each reactor data is classified as a [Variable](variable_class.md). Since the main scope of `fusdb` is to infer missing data from a reactor and find inconsistencies, the variables are considered as "suggestions" rather than being fixed (unless specified in the reactor file).

A [Relation](relation_class.md) on the other hand is always considered true, as it represents a physical relation between a number of variables.

A set of relations and variables can be used to create a [RelationSystem](relationsystem_class.md), that handles the connections between relations and variables and hosts the solver. 

The full set of relations and variables is represented by the following graph:
<div style="width: 100%; height: 900px; border: 1px solid #e1e4e5;">
  <iframe src="../relations_variables_graph.html" style="width: 100%; height: 100%; border: 0;" loading="lazy"></iframe>
</div>
