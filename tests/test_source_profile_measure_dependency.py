import numpy as np
import pytest

from fusdb.profile_system import build_relation_system
from fusdb.utils.profiles import volume_average
from fusdb.variable import Variable


def test_movable_source_profile_promotes_available_volume_weight_to_graph_input():
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_profile = 1.0 + 2.0 * source_coordinate**2
    rho = np.linspace(0.0, 1.0, 46)
    rho_tor = rho**1.2
    weight = 0.2 + rho**2

    system = build_relation_system(
        [
            Variable("rho_tor", value=rho_tor),
            Variable("w_V", value=weight),
            Variable(
                "n_e",
                value=source_profile,
                coordinate="rho_tor",
                coordinate_values=source_coordinate,
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
    assert "w_V" in source_relation.input_names
    assert "w_V" not in source_relation.constant_names

    system.compile()
    values = system.complete(system.solver_values())
    average = float(np.asarray(values["n_e_avg"]).reshape(-1)[0])
    assert volume_average(values["n_e"], values["rho"], weight=values["w_V"]) == pytest.approx(average)
