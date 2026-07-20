from pathlib import Path

import numpy as np

import fusdb
import fusdb.plotting as plotting
from fusdb.plotting.data import Curve, CurveSet, FieldMap, TableCell, TableData
from fusdb.plotting.tables import render_table
from fusdb.utils.datasets import PreparedTable


def test_curve_set_keeps_one_source_of_xy_data_for_both_backends() -> None:
    curve = Curve([1, 2], [3, 4], "demo", columns={"y_alt": [5, 6]}, metadata={"family": "test"})
    data = CurveSet([curve], xlabel="x", ylabel="y", xscale="log")

    assert data.curves[0].source_data()["y_alt"].tolist() == [5.0, 6.0]
    assert data.curves[0].metadata["family"] == "test"


def test_field_map_requires_aligned_fields() -> None:
    data = FieldMap([[0, 1], [0, 1]], [[0, 0], [1, 1]], {"value": [[1, 2], [3, 4]]})
    assert data.fields["value"].shape == (2, 2)


def test_table_data_renders_html_and_text() -> None:
    data = TableData(["case"], [("R", [TableCell("3.2", background="#c6efce")])])

    assert "background-color:#c6efce" in render_table(data)
    assert "R" in render_table(data, format="text")


def test_prepared_table_exposes_named_columns_for_explicit_curve_data() -> None:
    table = PreparedTable(
        path=Path("demo.yaml"),
        reaction_id="demo",
        metadata={},
        quantities=("temperature", "sigmav"),
        units=("kev", "m^3/s"),
        columns=(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
    )

    curves = CurveSet(
        [Curve(table.column("temperature"), table.column("sigmav"), "sigmav")],
        xlabel="temperature [kev]",
    )
    assert curves.curves[0].y.tolist() == [3.0, 4.0]


def test_legacy_plotting_and_table_aliases_are_not_public() -> None:
    for name in (
        "plot_curves", "bokeh_curves", "plot_profiles", "plot_profile_grid",
        "plot_reactivity", "plot_parameter_map", "plot_popcon", "variables_table",
    ):
        assert not hasattr(plotting, name)
    assert not hasattr(fusdb, "variables_table")
    assert not hasattr(fusdb.Reactor, "print_variables_table")
    assert not hasattr(fusdb.Reactor, "print_html_variables_table")
