from fusdb import CompilePlan, RelationSystem, Variable
from fusdb.registry import RELATIONS


def _model():
    return RelationSystem(
        [Variable("R", 6.0), Variable("a", 2.0)],
        [RELATIONS.get("Aspect ratio")],
        name="compile_plan_contract",
    )


def test_compile_returns_an_executable_plan():
    plan = _model().compile(fixed={"R", "a"})

    assert isinstance(plan, CompilePlan)
    result = plan.run("verify")
    assert result["mode"] == "verify"
    assert result["success"]


def test_same_model_can_compile_independent_scenarios():
    model = _model()
    plan_a = model.compile(inputs={"R": 6.0, "a": 2.0}, fixed={"R", "a"})
    plan_b = model.compile(inputs={"R": 9.0, "a": 3.0}, fixed={"R", "a"})

    assert plan_a is not plan_b
    assert plan_a.values["R"] == 6.0
    assert plan_b.values["R"] == 9.0
    assert plan_a.run("verify")["success"]
    assert plan_b.run("verify")["success"]

    plan_b.values["R"] = 12.0
    assert plan_a.values["R"] == 6.0
