from __future__ import annotations

import textwrap

import pytest

from fusdb import Reactor, SolvedColumn, Variable, aspect_ratio, render_table, solve_reactors, variable_table_data


def _reactor(name: str, radius: float) -> Reactor:
    return Reactor(
        name=name,
        variables={
            "R": Variable("R", value=radius),
        },
    )


def test_variable_table_data_renders_loaded_reactor_without_running_relations():
    reactor = _reactor("Example", 3.2)
    reactor.relations = lambda: (_ for _ in ()).throw(AssertionError("display should not select relations"))

    html = render_table(variable_table_data(reactor))

    assert "Example" in html
    assert "3.2" in html


def test_variable_table_data_combines_multiple_reactors():
    html = render_table(variable_table_data(_reactor("A", 3.2), _reactor("B", 4.4)))

    assert "A" in html
    assert "B" in html
    assert "3.2" in html
    assert "4.4" in html


def test_variable_table_data_can_display_existing_system_state():
    reactor = _reactor("Solved", 3.2)
    reactor.relations = lambda: ()
    system = reactor.relation_system().compile()
    system.values["R"] = 4.4

    html = render_table(variable_table_data(system, variable_names=("R",)))

    assert "Solved" in html
    assert "4.4" in html


def test_variable_table_data_renders_solved_column_snapshot():
    column = SolvedColumn(
        name="Snap",
        inputs={"R": 3.0},
        values={"R": 4.4},
        rel_tols={"R": 0.01},
        abs_tols={"R": 0.0},
        active_variable_names=frozenset({"R"}),
        relation_names_by_variable={"R": ("geometry rule",)},
        result={"success": True},
    )

    html = render_table(variable_table_data(column))

    assert "Snap" in html
    assert "#1EFF00" in html  # success-coloured header
    assert "title='geometry rule'" in html  # tooltip from the snapshot
    assert "4.4" in html


def test_run_absorbs_solved_values_and_keeps_last_plan():
    reactor = _reactor("Absorb", 3.2)
    reactor.relations = lambda: ()

    reactor.run("verify")

    assert reactor.last_plan is not None
    # Variable is immutable, so a run never rewrites the declaration; nothing
    # moves it here (verify performs no solve) so the read-through .value
    # still agrees with the declared value and with the solved system.
    assert reactor.R.value == reactor.last_plan.values["R"]
    assert reactor.R.declared.value == reactor.R.value


def test_reconcile_moves_value_without_touching_the_declaration():
    reactor = Reactor(
        name="Move",
        variables={
            "R": Variable("R", value=3.0, fixed=True),
            "a": Variable("a", value=1.0, fixed=True),
            # Declared inconsistent with R/a=3.0 on purpose; only A can move
            # (R and a are fixed), so reconcile is forced to correct A.
            "A": Variable("A", value=5.0),
        },
    )
    reactor.relations = lambda: (aspect_ratio,)

    result = reactor.reconcile()

    assert result["success"]
    # The declaration is exactly what was supplied, forever -- a solve does
    # not (and now cannot) rewrite it.
    assert reactor.get_variable("A").declared.value == 5.0
    # The read-through .value reflects the solved system instead, and it
    # genuinely differs from the declaration here.
    assert reactor.A.value == pytest.approx(3.0)
    assert reactor.A.value != reactor.A.declared.value
    assert reactor.A.value == reactor.last_plan.values["A"]
    # A fixed variable never moves: declared and solved agree trivially.
    assert reactor.R.value == reactor.R.declared.value == 3.0

    # The table path reads through _table_column's Reactor branch, which must
    # show the solved value (3), not the frozen declaration (5). A Reactor
    # column always reports no active variables, so the diff-colouring paths
    # never fire for it -- this only checks which value is displayed.
    html_after_solve = render_table(variable_table_data(reactor, variable_names=("A",)))
    assert ">3<" in html_after_solve
    assert ">5<" not in html_after_solve

    # A later run starts from the same declarations again, not the solution.
    second = reactor.reconcile()
    assert second["success"]
    assert reactor.get_variable("A").declared.value == 5.0

    # restart_from_solution() is the explicit opt-in to continue from here.
    reactor.restart_from_solution()
    assert reactor.get_variable("A").declared.value == pytest.approx(3.0)


def test_add_variable_rejects_a_solved_variable_view():
    reactor = _reactor("Guard", 3.2)
    reactor.relations = lambda: ()
    reactor.run("verify")

    with pytest.raises(TypeError):
        reactor.add_variable(reactor.get_variable("R"))


def _write_mini_reactor(directory, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reactor.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            metadata:
              name: {name}
            tags: ["__no_relations__"]
            variables:
              R: 3.0
              a: 1.0
            """
        ).strip()
    )
    return str(path)


def test_solve_reactors_runs_in_parallel_and_labels_duplicates(tmp_path):
    paths = [
        _write_mini_reactor(tmp_path / "alpha", "Mini"),
        _write_mini_reactor(tmp_path / "beta", "Mini"),
    ]

    columns = solve_reactors(paths, mode="verify", workers=2)

    assert all(isinstance(column, SolvedColumn) for column in columns)
    # Duplicate reactor names are disambiguated by their location.
    names = [column.name for column in columns]
    assert names == ["Mini (alpha/reactor.yaml)", "Mini (beta/reactor.yaml)"]
    assert render_table(variable_table_data(*columns)).startswith("<table")
