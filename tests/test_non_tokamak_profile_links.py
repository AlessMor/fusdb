from pathlib import Path

from fusdb import Reactor
from fusdb.registry import RELATIONS, TAGS


ROOT = Path(__file__).parents[1]
STELLARIS = ROOT / "reactors" / "STELLARIS" / "reactor.yaml"


def _selected_names(device: str) -> set[str]:
    return {
        relation.name
        for relation in RELATIONS.get_filtered_relations(tags=TAGS.expand((device,)))
    }


def test_reduced_non_tokamak_line_average_restores_legacy_provider():
    for device in ("stellarator", "mirror"):
        names = _selected_names(device)
        assert "Reduced non-tokamak electron density line-average" in names
        assert "Electron density line-average" not in names


def test_stellarator_profile_average_links_have_reduced_volume_measure():
    names = _selected_names("stellarator")
    assert "Reduced stellarator volume integration weight" in names
    for relation_name in (
        "Electron temperature volume-average consistency",
        "Ion temperature volume-average consistency",
        "Electron density volume-average consistency",
        "Ion density volume-average consistency",
    ):
        assert relation_name in names


def test_mirror_profile_average_links_have_reduced_volume_measure():
    names = _selected_names("mirror")
    assert "Reduced mirror volume integration weight" in names
    for relation_name in (
        "Electron temperature volume-average consistency",
        "Ion temperature volume-average consistency",
        "Electron density volume-average consistency",
        "Ion density volume-average consistency",
    ):
        assert relation_name in names


def test_stellaris_keeps_n_la_and_sudo_chain_active():
    reactor = Reactor.from_yaml(STELLARIS)
    system = reactor.relation_system().compile()
    active = {relation.name for relation in system.primary_relations}

    assert "Reduced stellarator volume integration weight" in active
    assert "Electron density volume-average consistency" in active
    assert "Reduced non-tokamak electron density line-average" in active
    assert "Sudo density limit" in active
    assert "Sudo density fraction" in active
    assert "Sudo margin" in active
