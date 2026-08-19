from __future__ import annotations

from collections.abc import Mapping

import fusdb
from fusdb.registry import DATASETS, REACTIONS, RELATIONS, RelationRegistry, get_relations, load_dataset
from fusdb.registry import dataset as dataset_module
from fusdb.registry import reaction_registry, relation_registry


def test_reaction_metadata_is_a_read_only_mapping() -> None:
    assert isinstance(REACTIONS, Mapping)
    assert REACTIONS["DT"].stoichiometry("D") == -1
    assert REACTIONS["DT"].stoichiometry("He4") == 1
    assert not hasattr(reaction_registry, "ReactionRegistry")


def test_dataset_index_and_loader_share_stable_ids() -> None:
    assert isinstance(DATASETS, Mapping)
    dataset_id = "reactivity_NRL_DT"
    assert dataset_id in DATASETS
    assert load_dataset(dataset_id).dataset_id == dataset_id
    assert not hasattr(dataset_module, "DatasetRegistry")


def test_relation_discovery_is_cached_without_a_proxy_class() -> None:
    registry = get_relations()
    assert registry is get_relations() is RELATIONS
    assert isinstance(registry, RelationRegistry)
    assert len(registry) > 0
    assert registry.get("aspect_ratio").function_name == "aspect_ratio"
    assert not hasattr(relation_registry, "LazyRelationRegistry")


def test_implementation_classes_are_not_top_level_api() -> None:
    for name in ("CompilePlan", "RelationRegistry", "SpeciesRegistry", "TagRegistry", "VariableRegistry"):
        assert not hasattr(fusdb, name)
