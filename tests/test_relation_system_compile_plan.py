from fusdb import CompilePlan, RelationSystem, Variable
from fusdb.registry import RELATIONS


def _model():
    return RelationSystem(
        [Variable("R", 6.0), Variable("a", 2.0)],
        [RELATIONS.get("Aspect ratio")],
        name="compile_plan_contract",
    )


def test_relation_system_is_lazy_reusable_model():
    model = _model()
    assert model._graph is None
    assert not hasattr(model, "pack")
    assert not hasattr(model, "run")

    plan = model.compile()
    assert isinstance(plan, CompilePlan)
    assert plan.model is model
    assert not hasattr(plan, "_graph")
    assert not hasattr(plan, "_structural_graph")
    assert not hasattr(plan, "compile")


def test_compile_plans_are_independent_and_do_not_mutate_model_graph():
    model = _model()
    plan_a = model.compile(fixed={"R"})
    plan_b = model.compile(inputs={"R": 9.0, "a": 3.0}, fixed={"R", "a"})

    assert plan_a is not plan_b
    before = dict(plan_a.values)
    plan_b.values["R"] = 10.0
    assert plan_a.values == before

    relation_node = ("relation", "Aspect ratio")
    assert model.graph.nodes[relation_node].get("active") is None
    assert "Aspect ratio" in plan_a.active_relations


def test_compile_plan_runs_without_recompiling_model():
    model = _model()
    plan = model.compile(fixed={"R", "a"})
    result = plan.run("verify")
    assert result["mode"] == "verify"
    assert result["success"]


def test_matching_scenarios_reuse_model_structure_cache(monkeypatch):
    model = _model()
    model.compile()
    calls = 0
    original = CompilePlan._run_compile_pass

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(CompilePlan, "_run_compile_pass", counted)
    second = model.compile(inputs={"R": 9.0, "a": 3.0})

    assert calls == 0
    assert len(model._structure_cache) == 1
    assert second.values["R"] == 9.0
