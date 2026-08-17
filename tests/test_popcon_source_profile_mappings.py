"""Worker reconstruction for source profiles on explicit geometry mappings."""

from __future__ import annotations

import numpy as np

from fusdb.modes import popcon as popcon_mode
from fusdb.profile_system import build_relation_system
from fusdb.registry import RELATIONS
from fusdb.variable import Variable


MAPPING = "Tokamak normalized minor-radius coordinate"
RHO_AVERAGE = "Electron density rho-average"


def _mapped_source_system(*, fixed: bool = False, profile_size: int = 31):
    source_coordinate = np.linspace(0.0, 1.0, 101)
    source_values = 3.0e19 * (1.0 - 0.45 * source_coordinate**2)
    return build_relation_system(
        [
            Variable(
                "n_e",
                value=source_values,
                coordinate="rho_minor",
                coordinate_values=source_coordinate,
                fixed=fixed,
            ),
            Variable("density_peaking", value=1.5),
        ],
        (RELATIONS.get(MAPPING), RELATIONS.get(RHO_AVERAGE)),
        profile_size=profile_size,
    ).compile()


def test_worker_rebuild_preserves_foreign_coordinate_source_profile():
    original = _mapped_source_system(profile_size=31)
    spec = popcon_mode._system_spec(original)

    assert any(
        item.get("kind") == "source_profile" and item.get("coordinate") == "rho_minor"
        for item in spec["relations"]
    )

    # Geometry-independent tokamak coordinate defaults are materialized by the
    # source-aware builder and removed from the solver graph. Worker rebuilds
    # must therefore preserve the resulting fixed mapping as serialized input
    # data rather than require the original fallback provider relation.
    serialized = {name: (value, fixed) for name, value, fixed, *_ in spec["variables"]}
    assert "rho_minor" in serialized
    serialized_mapping, mapping_fixed = serialized["rho_minor"]
    assert mapping_fixed

    rebuilt = popcon_mode._rebuild_system(spec).compile()
    original_values = original.complete(dict(original.input_values()))
    rebuilt_values = rebuilt.complete(dict(rebuilt.input_values()))

    assert rebuilt.profile_size == 31
    np.testing.assert_array_equal(serialized_mapping, original_values["rho_minor"])
    np.testing.assert_array_equal(rebuilt_values["rho_minor"], original_values["rho_minor"])
    np.testing.assert_allclose(rebuilt_values["n_e"], original_values["n_e"], rtol=0.0, atol=0.0)


def test_fixed_foreign_coordinate_source_remains_absolute_after_worker_rebuild():
    original = _mapped_source_system(fixed=True, profile_size=31)
    rebuilt = popcon_mode._rebuild_system(popcon_mode._system_spec(original)).compile()

    original_values = original.complete(dict(original.input_values()))
    rebuilt_values = rebuilt.complete(dict(rebuilt.input_values()))

    assert "n_e_avg" not in rebuilt.inputs
    np.testing.assert_allclose(rebuilt_values["n_e"], original_values["n_e"], rtol=0.0, atol=0.0)


def test_parallel_popcon_with_foreign_coordinate_source_matches_serial():
    x = {"variable": "n_e_avg", "values": [2.0e19, 3.0e19]}
    y = {"variable": "density_peaking", "values": [1.2, 1.8]}
    outputs = ("n_e_rho_avg",)

    serial = popcon_mode.run(_mapped_source_system(), x=x, y=y, outputs=outputs)
    parallel = popcon_mode.run(
        _mapped_source_system(),
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
