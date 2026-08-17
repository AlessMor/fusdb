import numpy as np

from fusdb import build_relation_system
from fusdb.variable import Variable


def test_public_builder_uses_canonical_grid_for_external_source_profile():
    source = np.linspace(0.0, 1.0, 101)
    variables = [
        Variable("B", value=np.linspace(0.0, 1.0, 46), fixed=True),
        Variable(
            "n_e",
            value=1.0e20 * (1.0 - 0.3 * source**2),
            coordinate="B",
            coordinate_values=source,
        ),
    ]

    system = build_relation_system(variables, (), profile_size=46, name="source_builder")
    system.compile()
    completed = system.complete(system.solver_values())

    assert system.profile_size == 46
    assert np.asarray(completed["n_e"]).shape == (46,)
    assert completed.get("n_e_avg") is not None
