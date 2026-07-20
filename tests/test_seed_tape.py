"""Seeding-tape replay must be bit-identical to fresh discovery.

The tape records the oracle's discovery steps at a full compile so a later
compile with the same structure but new values can replay them instead of
re-discovering (see RelationSystem.compile / _replay_seed_tape).  The safety
invariant is bit-identity: replay must produce exactly the seeds fresh
discovery would at the same values -- a nearby-but-different seed shifts
certified popcon values.  These cases pin the two tape branches the reactor
fixtures do not exercise (their tapes are all ``invert``/``block``): the
multi-output ``forward`` step and the registry ``default`` step.
"""

from __future__ import annotations

import numpy as np

from fusdb import RelationSystem, Variable, relation
from fusdb.registry import RELATIONS
from fusdb.relationsystem import _replay_seed_tape, initial_values_from_graph


def _assert_replay_matches_discovery(system: RelationSystem, perturb: dict[str, float]) -> None:
    for name, value in perturb.items():
        system.inputs[name] = value
        system.values[name] = value
    replayed = _replay_seed_tape(system)
    assert replayed is not None, "replay fell back to discovery"
    replay_seeds, replay_prov = replayed
    discovery_seeds, discovery_prov = initial_values_from_graph(system)
    assert set(replay_seeds) == set(discovery_seeds)
    assert replay_prov == discovery_prov
    for name, expected in discovery_seeds.items():
        np.testing.assert_array_equal(
            np.asarray(replay_seeds[name], dtype=float),
            np.asarray(expected, dtype=float),
            err_msg=f"replay seed {name!r} differs from discovery",
        )


def test_forward_step_replays_bit_identically() -> None:
    @relation(outputs=("A", "a"), name="split_forward_probe")
    def split_forward_probe(R):  # one input, two outputs -> forward step
        return {"A": R / 1.0, "a": R * 2.0}

    system = RelationSystem(
        [Variable("R", 3.0), Variable("A"), Variable("a")], [split_forward_probe]
    )
    system.compile()
    assert any(step[0] == "forward" for step in system._seed_tape)

    _assert_replay_matches_discovery(system, {"R": 4.5})


def test_registry_default_step_replays_bit_identically() -> None:
    # tau_p supplied makes f_D a gated free-core default; n_D unknown blocks any
    # forward derivation, so f_D is seeded by _seed_defaults -> a "default" step.
    rel = RELATIONS.get("deuterium_density_from_ion_density_and_fraction")
    system = RelationSystem(
        [Variable("n_i", 1.0e20), Variable("tau_p", 1.0), Variable("f_D"), Variable("n_D")],
        [rel],
    )
    system.compile()
    assert any(step[0] == "default" for step in system._seed_tape)
    assert system.seed_provenance.get("f_D") == "registry_default"

    _assert_replay_matches_discovery(system, {"n_i": 1.2e20})
