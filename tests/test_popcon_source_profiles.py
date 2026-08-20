"""POPCON source-profile behavior through the public scan interface."""

import numpy as np

from fusdb.modes import popcon as popcon_mode
from fusdb.profiles.system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.variable import Variable


def _source_system(profile_size: int = 31):
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_values = 3.0e19 * (1.0 - 0.45 * source_coordinate**2)
    return build_relation_system(
        [
            Variable(
                "n_e",
                value=source_values,
                coordinate="rho",
                coordinate_values=source_coordinate,
            ),
            Variable("density_peaking", value=1.5),
        ],
        (RELATIONS.get("Electron density rho-average"),),
        profile_size=profile_size,
    ).compile()


def test_parallel_popcon_with_source_profile_matches_serial():
    """Runtime source profiles must give the same scan result in serial and parallel."""
    x = {"variable": "n_e_avg", "values": [2.0e19, 3.0e19]}
    y = {"variable": "density_peaking", "values": [1.2, 1.8]}
    outputs = ("n_e_rho_avg",)

    serial = popcon_mode.run(_source_system(), x=x, y=y, outputs=outputs)
    parallel = popcon_mode.run(
        _source_system(), x=x, y=y, outputs=outputs, workers=2, chunk_size=2
    )

    assert serial["success"] and parallel["success"]
    serial_payload = serial["popcon"]
    parallel_payload = parallel["popcon"]
    assert np.array_equal(parallel_payload["success"], serial_payload["success"])
    np.testing.assert_allclose(
        parallel_payload["fields"]["n_e_rho_avg"],
        serial_payload["fields"]["n_e_rho_avg"],
        rtol=1.0e-12,
        atol=0.0,
    )
