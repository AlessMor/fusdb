from fusdb import RelationSystem, Variable
from fusdb.relationsystem import CompilePlan
from fusdb.registry import RELATIONS


def _model():
    return RelationSystem(
        [Variable("R", 6.0), Variable("a", 2.0)],
        [RELATIONS.get("Aspect ratio")],
        name="compile_plan_contract",
    )


def test_compile_produces_independent_executable_scenarios():
    """Compiling the same model twice must produce independent runnable scenarios."""
    model = _model()
    plan_a = model.compile(fixed={"R"})
    plan_b = model.compile(inputs={"R": 9.0, "a": 3.0}, fixed={"R", "a"})

    assert isinstance(plan_a, CompilePlan)
    assert isinstance(plan_b, CompilePlan)
    assert plan_a is not plan_b

    before = dict(plan_a.values)
    plan_b.values["R"] = 10.0
    assert plan_a.values == before

    result = plan_a.run("reconcile")
    assert result["mode"] == "reconcile"
    assert result["success"]
    assert result["values"]["A"] == 3.0


def test_fully_specified_compiled_scenario_verifies():
    plan = _model().compile(inputs={"R": 9.0, "a": 3.0, "A": 3.0}, fixed={"R", "a", "A"})
    result = plan.run("verify")

    assert result["mode"] == "verify"
    assert result["success"]
