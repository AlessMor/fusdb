from fusdb.profile_system import build_relation_system
from fusdb.registry import RELATIONS


def test_tokamak_coordinate_residual_growth_is_domain_enforcement_only():
    """Account explicitly for the five rows introduced by coordinate domains.

    The three reduced coordinate providers are structural completion relations,
    not nonlinear equations, so they add no enforced relation residuals and no
    solver DOFs.  Their physical domains do add five hard feasibility rows:
    lower+upper bounds for rho_minor and v_norm, and the non-negative lower
    bound for w_V.  This is intentional -- variable domains remain enforced
    boundaries -- and explains the observed +5 residual_size without hiding an
    extra physics constraint.
    """
    relations = [
        RELATIONS.get("Tokamak normalized minor-radius coordinate"),
        RELATIONS.get("Tokamak normalized enclosed volume"),
        RELATIONS.get("Tokamak volume integration weight"),
    ]
    system = build_relation_system([], relations, profile_size=46)
    system.compile()
    system.pack()
    values = system.complete(system.solver_values())
    layout = system.residual_layout(values)

    assert system.packed_dim == 0
    assert system._enforced_residual_relations == []
    assert layout["relation_dims"] == []
    assert layout["size"] == 5
    assert {name for name, *_rest in layout["domain_tail"]} == {
        "rho_minor",
        "v_norm",
        "w_V",
    }
