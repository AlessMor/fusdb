from pathlib import Path

import pytest

from fusdb import Reactor


REACTORS = Path(__file__).parents[1] / "reactors"


@pytest.mark.parametrize("name", ["DEMO_2022", "ARC_V0", "Polomac"])
def test_representative_reactors_load_and_compile(name):
    """Shipped reactor YAMLs should survive the full ingestion/compile boundary."""
    reactor = Reactor.from_yaml(REACTORS / name)
    plan = reactor.relation_system().compile()

    assert plan is not None


def test_reactor_model_can_be_compiled_repeatedly():
    model = Reactor.from_yaml(REACTORS / "DEMO_2022").relation_system()
    first = model.compile()
    second = model.compile()

    assert first is not second
