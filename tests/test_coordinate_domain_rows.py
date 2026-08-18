from fusdb.profiles.system import build_relation_system
from fusdb.registry import RELATIONS


def test_static_tokamak_coordinate_defaults_add_no_solver_domain_rows():
    """Static fallback mappings are framework data, not solver constraints.

    The identity/self-similar tokamak defaults are evaluated once by the
    source-aware builder and materialized as fixed coordinate data. They carry
    no solver ancestry, so keeping their registry domains in the nonlinear
    residual would add five identically satisfied rows and change finite-
    difference grouping without constraining any unknown. Their values remain
    validated when they are constructed; supplied or geometry-derived mappings
    stay ordinary active variables and retain domain enforcement.
    """
    relations = [
        RELATIONS.get("Tokamak normalized minor-radius coordinate"),
        RELATIONS.get("Tokamak normalized enclosed volume"),
        RELATIONS.get("Tokamak volume integration weight"),
    ]
    system = build_relation_system([], relations, profile_size=46).compile()
    system.pack()
    values = system.complete(system.solver_values())
    layout = system.residual_layout(values)

    assert system.packed_dim == 0
    assert system.residual_relations == []
    assert layout["relation_dims"] == []
    assert layout["size"] == 0
    assert layout["domain_tail"] == []


def test_geometry_derived_coordinate_keeps_domain_rows():
    """A real geometry mapping remains inside the enforced solver domain."""
    relation = RELATIONS.get("Sauter self-similar profile volume mapping")
    system = build_relation_system(
        [],
        [relation],
        profile_size=46,
    ).compile()
    # The relation inputs are deliberately absent here: compilation may prune
    # the provider, but the builder must not have folded it into fixed data.
    assert relation.name in {item.name for item in system.model.candidate_primary_relations}
    assert system.inputs.get("v_norm") is None
    assert system.inputs.get("w_V") is None
