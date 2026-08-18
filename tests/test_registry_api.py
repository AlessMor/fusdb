from __future__ import annotations

from collections.abc import Mapping

from fusdb.registry import DATASETS, REACTIONS, get_relations, load_dataset


def test_reaction_metadata_is_a_read_only_mapping() -> None:
    assert isinstance(REACTIONS, Mapping)
    assert REACTIONS["DT"].stoichiometry("D") == -1
    assert REACTIONS["DT"].stoichiometry("He4") == 1


def test_dataset_index_and_loader_share_stable_ids() -> None:
    assert isinstance(DATASETS, Mapping)
    dataset_id = "reactivity_NRL_DT"
    assert dataset_id in DATASETS
    assert load_dataset(dataset_id).dataset_id == dataset_id


def test_relation_discovery_is_cached_and_functional() -> None:
    registry = get_relations()
    assert registry is get_relations()
    assert len(registry) > 0
    assert registry.get("aspect_ratio").function_name == "aspect_ratio"
