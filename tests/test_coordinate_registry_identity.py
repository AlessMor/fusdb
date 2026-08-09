from fusdb.registry import VARIABLES
from fusdb.registry.variable_registry import VARIABLES as DIRECT_VARIABLES


def test_coordinate_registry_overlay_preserves_singleton_identity():
    assert VARIABLES is DIRECT_VARIABLES
    assert DIRECT_VARIABLES.resolve("normalized_minor_radius") == "rho_minor"
    assert DIRECT_VARIABLES.resolve("r_over_a") == "rho_minor"
    assert "w_V" in DIRECT_VARIABLES
