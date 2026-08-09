from fusdb.registry import RELATIONS


def test_process_pedestal_profiles_use_explicit_minor_radius_mapping():
    names = (
        "PROCESS pedestal electron temperature profile",
        "PROCESS pedestal ion temperature profile",
        "PROCESS pedestal electron density profile",
        "PROCESS pedestal fuel-ion density profile",
    )
    for name in names:
        relation = RELATIONS.get(name)
        assert "rho_minor" in relation.input_names
        assert "rho" not in relation.input_names
        assert "rho" not in relation.constant_names
