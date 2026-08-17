import numpy as np

from fusdb.profile_system import build_relation_system
from fusdb.variable import Variable


def test_fixed_source_profile_does_not_gain_volume_measure_dependency():
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_profile = 2.0 + source_coordinate
    rho = np.linspace(0.0, 1.0, 46)

    system = build_relation_system(
        [
            Variable("rho_tor", value=rho**1.1),
            Variable("w_V", value=rho),
            Variable(
                "n_e",
                value=source_profile,
                coordinate="rho_tor",
                coordinate_values=source_coordinate,
                fixed=True,
            ),
        ],
        [],
        profile_size=46,
    )
    source_relation = next(
        relation
        for relation in system.candidate_primary_relations
        if relation.source_kind == "source_profile" and relation.source_name == "n_e"
    )

    assert "rho_tor" in source_relation.input_names
    assert "w_V" not in source_relation.input_names
    assert "v_norm" not in source_relation.input_names
