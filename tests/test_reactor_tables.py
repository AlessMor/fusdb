"""Tests for reactor comparison table payload and rendering helpers."""

from __future__ import annotations

from fusdb.plotting.tables import build_reactor_comparison_table
from fusdb.reactor_class import Reactor
from fusdb.relation_class import Relation
from fusdb.variable_class import Variable


def _explicit_scalar(name: str, value: float):
    """Return one explicit scalar variable.

    Args:
        name: Variable name.
        value: Explicit input value.

    Returns:
        Variable object with explicit input state.
    """
    # Create one scalar variable and seed explicit input/current value.
    var = Variable.make(name=name, ndim=0)
    var.add_value(value, as_input=True)
    var.input_source = "explicit"
    return var


def _empty_scalar(name: str):
    """Return one scalar variable with no input/current value."""
    return Variable.make(name=name, ndim=0)


def test_reactor_table_payload_includes_diagnostic_status() -> None:
    """Expected: ``to_table_payload`` carries variable diagnostic status."""
    # Build one violated relation where x should be 2 while input is 1.
    relation = Relation.from_callable(
        name="test_constant_x",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )
    reactor = Reactor(
        id="R_test",
        variables_dict={"x": _explicit_scalar("x", 1.0)},
        relations=[relation],
    )

    # Build one payload and assert diagnostic status is available.
    payload = reactor.to_table_payload(include_diagnostics=True)
    variable_payload = payload["variables"]["x"]
    assert variable_payload["diag_status"] == "INCONSISTENT"
    assert (variable_payload["diag_rank"] or 0) >= 1


def test_reactor_comparison_table_formats_inconsistent_values() -> None:
    """Expected: inconsistent explicit values render as ``input -> computed``."""
    # Build one violated relation where x should be 2 while input is 1.
    relation = Relation.from_callable(
        name="test_constant_x",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )
    reactor = Reactor(
        id="R_test",
        variables_dict={"x": _explicit_scalar("x", 1.0)},
        relations=[relation],
    )
    reactor.solve()

    # Build table and assert inconsistent display format contract.
    table = build_reactor_comparison_table({"R_test": reactor})
    cell_text = str(table["dataframe"].loc["x", "R_test"])
    assert "->" in cell_text
    assert "<u>" not in cell_text and "</u>" not in cell_text
    assert "2" in cell_text
    assert "1.0 -> 1.0" not in cell_text


def test_reactor_comparison_table_is_snapshot_and_does_not_auto_solve() -> None:
    """Expected: table rendering does not mutate reactor values by re-running solve."""
    relation = Relation.from_callable(
        name="test_constant_x",
        target="x",
        inputs=(),
        func=lambda: 2.0,
    )
    reactor = Reactor(
        id="R_snapshot",
        variables_dict={"x": _explicit_scalar("x", 1.0)},
        relations=[relation],
    )

    table = build_reactor_comparison_table({"R_snapshot": reactor})
    cell_text = str(table["dataframe"].loc["x", "R_snapshot"])

    # Snapshot keeps input/current unchanged when solve() was not called.
    assert "->" not in cell_text
    assert reactor.variables_dict["x"].current_value == 1.0


def test_reactor_comparison_table_uses_registry_variable_order() -> None:
    """Expected: variable rows follow allowed variable registry order."""
    reactor = Reactor(
        id="R_order",
        variables_dict={
            "a": _explicit_scalar("a", 1.0),
            "R": _explicit_scalar("R", 3.0),
        },
        relations=[],
    )

    table = build_reactor_comparison_table({"R_order": reactor}, include_diagnostics=False)
    rows = list(table["dataframe"].index)
    assert rows.index("R") < rows.index("a")


def test_reactor_comparison_table_displays_tags_metadata_row() -> None:
    """Expected: default metadata rows include reactor tags rendered as readable text."""
    reactor = Reactor(
        id="R_tags",
        tags=["demo", "tokamak"],
        variables_dict={"R": _explicit_scalar("R", 3.0)},
        relations=[],
    )

    table = build_reactor_comparison_table({"R_tags": reactor}, include_diagnostics=False)
    assert "tags" in table["dataframe"].index
    assert str(table["dataframe"].loc["tags", "R_tags"]) == "demo, tokamak"


def test_reactor_comparison_table_displays_solve_status_row() -> None:
    """Expected: default metadata rows include solve_status."""
    reactor = Reactor(
        id="R_status_unsolved",
        variables_dict={"R": _explicit_scalar("R", 3.0)},
        relations=[],
    )

    table = build_reactor_comparison_table({"R_status_unsolved": reactor}, include_diagnostics=False)
    assert "solve_status" in table["dataframe"].index
    assert str(table["dataframe"].loc["solve_status", "R_status_unsolved"]) == "not_solved"


def test_reactor_comparison_table_shows_final_check_stop_reason() -> None:
    """Expected: solve_status row reports final-check-specific stop reason after solve."""
    relation = Relation.from_callable(
        name="test_x_from_y",
        target="x",
        inputs=("y",),
        func=lambda y: y,
    )
    reactor = Reactor(
        id="R_status_solved",
        variables_dict={"x": _empty_scalar("x"), "y": _empty_scalar("y")},
        relations=[relation],
    )
    reactor.solve()

    table = build_reactor_comparison_table({"R_status_solved": reactor}, include_diagnostics=False)
    solve_status = str(table["dataframe"].loc["solve_status", "R_status_solved"])
    assert "final_check_undecidable" in solve_status
    assert "1 undecidable" in solve_status
