from types import SimpleNamespace

import numpy as np
import yaml

from fusdb.reactor import Reactor
from fusdb.variable import Variable


def test_reactor_yaml_reads_two_column_source_profile_without_forcing_grid_size(tmp_path):
    source = np.linspace(0.0, 1.0, 101)
    values = 1.0e20 * (1.0 - 0.25 * source**2)
    np.savetxt(tmp_path / "ne.dat", np.column_stack([source, values]))
    (tmp_path / "reactor.yaml").write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "source-yaml"},
                "grid": {"size": 46},
                "variables": {
                    "n_e": {
                        "file": "ne.dat",
                        "coordinate": "rho",
                    }
                },
            }
        )
    )

    reactor = Reactor.from_yaml(tmp_path)
    declared = reactor.variables["n_e"]

    assert declared.size == 101
    assert declared.coordinate == "rho"
    assert np.allclose(declared.coordinate_values, source)
    assert np.allclose(declared.value, values)

    system = reactor.relation_system().compile()
    assert system.profile_size == 46
    assert any(rel.source_kind == "source_profile" and "n_e" in rel.output_names for rel in system.model.candidate_primary_relations)


def test_reactor_yaml_passes_variable_local_default_relation(tmp_path):
    (tmp_path / "reactor.yaml").write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "provider-yaml"},
                "tags": ["tokamak"],
                "variables": {
                    "V_p": {"default_relation": "Plasma volume from arcs"},
                },
            }
        )
    )

    reactor = Reactor.from_yaml(tmp_path)
    names = {relation.name for relation in reactor.relations()}

    assert "Plasma volume from arcs" in names
    assert "Tokamak plasma volume" not in names


def test_programmatic_reactor_passes_local_provider_override():
    reactor = Reactor(
        name="provider-programmatic",
        tags=("tokamak",),
        variables={"V_p": Variable("V_p", default_relation="Plasma volume from arcs")},
    )

    names = {relation.name for relation in reactor.relations()}
    assert "Plasma volume from arcs" in names
    assert "Tokamak plasma volume" not in names


def test_restart_from_solution_turns_source_profile_into_canonical_snapshot():
    source = np.linspace(0.0, 1.0, 101)
    reactor = Reactor(
        name="restart-source",
        grid_size=46,
        variables={
            "n_e": Variable(
                "n_e",
                value=1.0e20 * (1.0 - 0.2 * source**2),
                coordinate="rho",
                coordinate_values=source,
            )
        },
    )
    reactor.last_plan = SimpleNamespace(values={"n_e": np.full(46, 8.0e19)})

    reactor.restart_from_solution()
    restarted = reactor.variables["n_e"]

    assert restarted.coordinate is None
    assert restarted.coordinate_values is None
    assert restarted.size == 46
    assert np.allclose(restarted.value, 8.0e19)
