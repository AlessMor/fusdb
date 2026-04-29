from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.reactor_class import Reactor
from fusdb.relationsystem_class import RelationSystem


def test_reactor_defers_relation_ordering_to_relationsystem() -> None:
    """Expected: Reactor stores YAML settings while RelationSystem builds ordered relations."""
    reactor = Reactor.from_yaml("reactors/DEMO_2022/reactor.yaml")
    system = reactor.make_relationsystem(mode="overwrite")

    assert reactor.relations == []
    assert not hasattr(reactor, "refresh_relations")
    assert system.relations


def test_relationsystem_template_apis_are_removed_and_construction_is_single_path() -> None:
    """Expected: template APIs are absent and Reactor.make_relationsystem accepts no template path."""
    reactor = Reactor.from_yaml("reactors/DEMO_2022/reactor.yaml")
    system = reactor.make_relationsystem(mode="overwrite")

    assert isinstance(system, RelationSystem)
    assert not hasattr(RelationSystem, "build_repo_template")
    assert not hasattr(RelationSystem, "from_template")
    assert not hasattr(RelationSystem, "export_relation_graph")
    with pytest.raises(TypeError):
        reactor.make_relationsystem(mode="overwrite", template=object())
    with pytest.raises(TypeError):
        reactor.make_relationsystem(mode="overwrite", enforce_constraint_tags=("constraint",))
    with pytest.raises(TypeError):
        reactor.solve(enforce_constraint_names=("greenwald_margin",))
