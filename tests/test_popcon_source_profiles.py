"""Parallel POPCON reconstruction of runtime-generated source profiles."""

from __future__ import annotations

import pickle

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


def test_source_profile_worker_recipe_is_picklable_and_preserves_grid():
    original = _source_system(profile_size=31)
    spec = popcon_mode._system_spec(original)

    pickle.dumps(spec)
    generated = [item for item in spec["relations"] if not isinstance(item, str)]
    assert len(generated) == 1
    assert generated[0].source_kind == "source_profile"
    assert generated[0].source_name == "n_e"
    assert any(item == "Electron density rho-average" for item in spec["relations"])

    rebuilt = popcon_mode._rebuild_system(spec).compile()
    assert rebuilt.profile_size == 31

    original_values = original.complete(dict(original.input_values()))
    rebuilt_values = rebuilt.complete(dict(rebuilt.input_values()))
    np.testing.assert_allclose(rebuilt_values["n_e"], original_values["n_e"], rtol=0.0, atol=0.0)
    assert np.asarray(rebuilt_values["n_e"]).shape == (31,)


def test_worker_reuses_prepared_model_for_matching_recipe():
    spec = popcon_mode._system_spec(_source_system(profile_size=31))
    popcon_mode._WORKER_MODELS.clear()

    for _ in range(2):
        popcon_mode._solve_batched_cases_from_spec(
            spec, "n_e_avg", "density_peaking", (), (), set(), ()
        )

    assert len(popcon_mode._WORKER_MODELS) == 1


def test_parallel_popcon_with_source_profile_matches_serial():
    x = {"variable": "n_e_avg", "values": [2.0e19, 3.0e19]}
    y = {"variable": "density_peaking", "values": [1.2, 1.8]}
    outputs = ("n_e_rho_avg",)

    serial = popcon_mode.run(_source_system(), x=x, y=y, outputs=outputs)
    parallel = popcon_mode.run(
        _source_system(),
        x=x,
        y=y,
        outputs=outputs,
        workers=2,
        chunk_size=2,
    )

    assert serial["success"] and parallel["success"]
    serial_payload = serial["popcon"]
    parallel_payload = parallel["popcon"]
    assert np.array_equal(parallel_payload["success"], serial_payload["success"])
    assert parallel_payload["success"].all()
    np.testing.assert_allclose(
        parallel_payload["fields"]["n_e_rho_avg"],
        serial_payload["fields"]["n_e_rho_avg"],
        rtol=1.0e-12,
        atol=0.0,
    )
