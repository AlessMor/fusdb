from __future__ import annotations

import textwrap

from fusdb import Reactor, SolvedColumn, Variable, solve_reactors, variables_table


def _reactor(name: str, radius: float) -> Reactor:
    return Reactor(
        name=name,
        variables={
            "R": Variable("R", value=radius),
        },
    )


def test_variables_table_renders_loaded_reactor_without_running_relations():
    reactor = _reactor("Example", 3.2)
    reactor.relations = lambda: (_ for _ in ()).throw(AssertionError("display should not select relations"))

    html = variables_table(reactor)

    assert "Example" in html
    assert "3.2" in html


def test_variables_table_combines_multiple_reactors():
    html = variables_table(_reactor("A", 3.2), _reactor("B", 4.4))

    assert "A" in html
    assert "B" in html
    assert "3.2" in html
    assert "4.4" in html


def test_variables_table_can_display_existing_system_state():
    reactor = _reactor("Solved", 3.2)
    reactor.relations = lambda: ()
    system = reactor.relation_system()
    system.variables_by_name["R"].set_value(4.4)

    html = variables_table(system, variable_names=("R",))

    assert "Solved" in html
    assert "4.4" in html


def test_variables_table_renders_solved_column_snapshot():
    var = Variable("R", value=3.0)
    var.set_value(4.4)  # input_value stays 3.0, value becomes 4.4
    column = SolvedColumn(
        name="Snap",
        variables_by_name={"R": var},
        active_variable_names=frozenset({"R"}),
        relation_names_by_variable={"R": ("geometry rule",)},
        result={"success": True},
    )

    html = variables_table(column)

    assert "Snap" in html
    assert "#1EFF00" in html  # success-coloured header
    assert "title='geometry rule'" in html  # tooltip from the snapshot
    assert "4.4" in html


def test_run_absorbs_solved_values_and_keeps_last_system():
    reactor = _reactor("Absorb", 3.2)
    reactor.relations = lambda: ()

    reactor.run("verify")

    assert reactor.last_system is not None
    # The solved value replaces the input entirely: value == input_value, and
    # both mirror the solved system.
    assert reactor.R.value == reactor.last_system.variables_by_name["R"].value
    assert reactor.R.input_value == reactor.R.value


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
    assert variables_table(*columns).startswith("<table")
