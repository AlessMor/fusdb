from fusdb.registry import RELATIONS, TAGS


def _names(**kwargs):
    return {rel.name for rel in RELATIONS.get_filtered_relations(tags=TAGS.expand(("tokamak",)), **kwargs)}


def test_scenario_default_relation_replaces_registry_preference():
    names = _names(default_relations={"V_p": ("Plasma volume from arcs",)})
    assert "Plasma volume from arcs" in names
    assert "Tokamak plasma volume" not in names


def test_scenario_default_relation_can_keep_simultaneous_providers():
    names = _names(
        default_relations={
            "V_p": ("Tokamak plasma volume", "Plasma volume from arcs"),
        }
    )
    assert "Tokamak plasma volume" in names
    assert "Plasma volume from arcs" in names


def test_empty_scenario_default_relation_removes_provider_gate():
    names = _names(default_relations={"V_p": ()})
    assert "Tokamak plasma volume" in names
    assert "Plasma volume from arcs" in names


def test_override_relation_must_produce_selected_variable():
    try:
        _names(default_relations={"V_p": ("Tokamak plasma surface",)})
    except ValueError as exc:
        assert "does not produce" in str(exc)
    else:
        raise AssertionError("invalid variable-local provider should fail")
