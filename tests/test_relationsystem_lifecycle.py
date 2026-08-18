from pathlib import Path

import pytest

from fusdb import Reactor


REACTORS = Path(__file__).parents[1] / "reactors"


def _reactor_dirs():
    return sorted(path for path in REACTORS.iterdir() if path.is_dir())


@pytest.mark.parametrize("reactor_dir", _reactor_dirs(), ids=lambda path: path.name)
def test_shipped_reactors_load_and_compile(reactor_dir):
    """Every shipped reactor scenario should remain usable after architecture refactors."""
    try:
        reactor = Reactor.from_yaml(reactor_dir)
        plan = reactor.relation_system().compile()
    except Exception as exc:
        pytest.fail(f"{reactor_dir.name} failed to load/compile: {exc}")

    assert reactor.name
    assert plan is not None
