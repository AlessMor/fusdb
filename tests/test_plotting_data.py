from pathlib import Path

import numpy as np
import pytest

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


def test_bokeh_curve_legend_is_a_compact_grid_below_the_figure() -> None:
    pytest.importorskip("bokeh")
    data = CurveSet([Curve([1, 2], [index, index + 1], f"curve {index}") for index in range(6)])

    plot, _, _ = plotting.bokeh_curve_set(data)
    legend = plot.legend[0]

    assert legend in plot.below
    assert legend not in plot.center
    assert legend.orientation == "horizontal"
    assert legend.ncols == len(data.curves)
    assert plot.frame_height == 620
    assert plot.height is None
    callbacks = plot.js_property_callbacks["change:inner_width"]
    assert len(callbacks) == 1
    assert "legend.ncols = best" in callbacks[0].code


def test_standard_bokeh_explorer_orders_plot_options_then_limits() -> None:
    pytest.importorskip("bokeh")
    from bokeh.models import CheckboxButtonGroup, LegendItem
    from bokeh.plotting import figure

    from fusdb.plotting._bokeh import explorer_layout

    plot = figure(width=600, height=400)
    renderer = plot.line([1, 2], [3, 4])
    selector = CheckboxButtonGroup(labels=["demo"], active=[0])
    layout, _status = explorer_layout(
        plot,
        legend_items=[LegendItem(label="demo", renderers=[renderer])],
        option_controls=[("Options", selector)],
        x_limits=(1, 2),
        y_limits=(3, 4),
    )

    assert layout.children[0] is plot
    assert "Options" in layout.children[1].children[0].text
    assert "Limits" in layout.children[2].children[0].text
    assert plot.frame_height == 400
    assert plot.height is None


def test_legend_columns_reflow_for_smaller_widths() -> None:
    from fusdb.plotting._bokeh import _columns_for_width

    item_widths = [100] * 6
    assert _columns_for_width(item_widths, 650, spacing=3) == 6
    assert _columns_for_width(item_widths, 250, spacing=3) == 2
    assert _columns_for_width(item_widths, 90, spacing=3) == 1


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
        "figure_to_html", "relation_graph_html",
    ):
        assert not hasattr(plotting, name)
    assert not hasattr(fusdb, "variables_table")
    assert not hasattr(fusdb.Reactor, "print_variables_table")
    assert not hasattr(fusdb.Reactor, "print_html_variables_table")


def test_thin_bokeh_html_and_save_wrappers_are_removed() -> None:
    from fusdb.plotting import atomic_physics, reactivity

    assert not hasattr(reactivity, "render_reactivity_app_html")
    assert not hasattr(reactivity, "save_reactivity_app_html")
    assert not hasattr(atomic_physics, "render_atomic_physics_app_html")
    assert not hasattr(atomic_physics, "save_atomic_physics_app_html")
