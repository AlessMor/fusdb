from fusdb.relation import Relation
from fusdb.registry import RELATIONS, TAGS
from fusdb.registry.relation_registry import RelationRegistry
from fusdb.registry.variable_registry import VariableRegistry, VariableSpec


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


def _synthetic_registry():
    variables = VariableRegistry(
        [
            VariableSpec("seed"),
            VariableSpec("x", default_relation=("x default",)),
            VariableSpec("y", default_relation=("y default",)),
        ]
    )
    x_default = Relation(
        name="x default",
        func=lambda seed: seed,
        input_names=("seed",),
        outputs=("x",),
        function_name="x_default_fn",
    )
    y_default = Relation(
        name="y default",
        func=lambda seed: seed,
        input_names=("seed",),
        outputs=("y",),
        function_name="y_default_fn",
    )
    multi = Relation(
        name="multi geometry",
        func=lambda seed: (seed, seed),
        input_names=("seed",),
        outputs=("x", "y"),
        function_name="multi_geometry_fn",
    )
    return variables, RelationRegistry((x_default, y_default, multi), variable_registry=variables)


def test_atomic_multi_output_override_replaces_registry_default_on_side_output():
    variables, relations = _synthetic_registry()
    names = {
        rel.name
        for rel in relations.get_filtered_relations(
            default_relations={"x": ("multi geometry",)},
            variable_registry=variables,
        )
    }
    assert names == {"multi geometry"}


def test_atomic_multi_output_override_rejects_incompatible_side_override():
    variables, relations = _synthetic_registry()
    try:
        relations.get_filtered_relations(
            default_relations={
                "x": ("multi geometry",),
                "y": ("y default",),
            },
            variable_registry=variables,
        )
    except ValueError as exc:
        assert "Multi-output relations are atomic" in str(exc)
    else:
        raise AssertionError("incompatible multi-output defaults should fail")


def test_empty_side_override_is_compatible_with_atomic_selection():
    variables, relations = _synthetic_registry()
    names = {
        rel.name
        for rel in relations.get_filtered_relations(
            default_relations={
                "x": ("multi geometry",),
                "y": (),
            },
            variable_registry=variables,
        )
    }
    assert "multi geometry" in names
    assert "x default" not in names
    assert "y default" in names
