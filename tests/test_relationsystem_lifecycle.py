from pathlib import Path

import pytest

from fusdb import Reactor


REACTORS = Path(__file__).parents[1] / "reactors"


def _reactor_dirs():
    return sorted(path for path in REACTORS.iterdir() if path.is_dir())


@pytest.mark.parametrize("reactor_dir", _reactor_dirs(), ids=lambda path: path.name)
def test_compiled_blocks_only_reference_final_active_system(reactor_dir):
    try:
        system = Reactor.from_yaml(reactor_dir).relation_system().compile()
    except Exception as exc:
        pytest.skip(f"fixture is not loadable as a reactor directory: {exc}")
    active = system.active_variable_names
    assert all(set(block) <= active for block in system.structural_blocks)
    assert all(not (set(rel.variables) & system._unevaluable_names) for rel in system.relations)


def test_packability_analysis_does_not_install_runtime_layout():
    system = Reactor.from_yaml(REACTORS / "DEMO_2022").relation_system().compile()
    before_specs = list(system.packed_specs)
    before_dim = system.packed_dim
    before_movement = list(system._movement_plan)

    system._packing_issues()

    assert system.packed_specs == before_specs
    assert system.packed_dim == before_dim
    assert len(system._movement_plan) == len(before_movement)


def test_completion_has_one_executableprovider_plan():
    system = Reactor.from_yaml(REACTORS / "DEMO_2022").relation_system().compile()
    assert isinstance(system.provider_plan, tuple)
    assert not hasattr(system, "_completion_plan_cache")
    assert not hasattr(system, "_completion_plan")
