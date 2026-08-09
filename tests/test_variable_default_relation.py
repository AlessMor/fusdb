from fusdb.registry import VARIABLES
from fusdb.variable import Variable


def test_variable_default_relation_inherits_registry_default():
    var = Variable("V_p")
    assert var.default_relation is None
    assert var.effective_default_relation == VARIABLES.get("V_p").default_relation


def test_variable_default_relation_can_override_registry_default():
    var = Variable("V_p", default_relation="PROCESS plasma volume")
    assert var.default_relation == ("PROCESS plasma volume",)
    assert var.effective_default_relation == ("PROCESS plasma volume",)


def test_variable_default_relation_preserves_multiple_simultaneous_relations():
    var = Variable(
        "V_p",
        default_relation=["Tokamak plasma volume", "PROCESS plasma volume", "Tokamak plasma volume"],
    )
    assert var.effective_default_relation == ("Tokamak plasma volume", "PROCESS plasma volume")


def test_variable_default_relation_empty_list_disables_registry_preference():
    var = Variable("V_p", default_relation=[])
    assert var.default_relation == ()
    assert var.effective_default_relation == ()
