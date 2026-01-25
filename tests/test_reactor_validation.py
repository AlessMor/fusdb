import math
from pathlib import Path

import pytest

from fusdb.relations.geometry import plasma_surface_area, plasma_volume
from fusdb.loader import load_reactor_yaml
from fusdb.reactor_class import Reactor
from fusdb import reactor_class as reactors_module
from fusdb.relation_class import Relation
from fusdb.relation_util import symbol


def test_missing_required_metadata_raises(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
            ]
        )
    )

    with pytest.raises(ValueError):
        load_reactor_yaml(path)


def test_optional_metadata_defaults_to_none(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Example Lab"',
            ]
        )
    )

    reactor = load_reactor_yaml(path)
    assert reactor.country is None
    assert reactor.design_year is None
    assert reactor.doi is None


def test_doi_can_be_list(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Example Lab"',
                "",
                "doi:",
                "  - 10.1234/one",
                "  - 10.1234/two",
            ]
        )
    )

    reactor = load_reactor_yaml(path)
    assert reactor.doi == ["10.1234/one", "10.1234/two"]


def test_parameters_missing_defaults(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Example Lab"',
                "",
                "# no parameters",
            ]
        )
    )

    reactor = load_reactor_yaml(path)
    assert reactor.parameters.get("P_fus") is None
    assert reactor.parameters.get("tau_E") is None


def test_aspect_ratio_and_minor_radius_are_backfilled(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
                [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 6.0",
                "A: 3.0",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.a), 2.0)
    assert math.isclose(float(reactor.A), float(reactor.R) / float(reactor.a))


def test_inconsistent_geometry_warns_and_updates_explicit(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
                [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 3.0",
                "a: 1.0",
                "A: 2.5",
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.A), 3.0)


def test_geometry_within_tolerance_no_warning(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
                [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 9.0",
                "a: 2.9",
                "A: 3.1",  # slightly off, but within 1%
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.A), 3.1)


def test_geometry_from_extents_and_kappa(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R_max: 5.0",
                "R_min: 3.0",
                "a: 1.0",
                "Z_max: 2.0",
                "Z_min: -2.0",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.R), 4.0)
    assert math.isclose(float(reactor.kappa), 2.0)


def test_method_override_selects_relation(monkeypatch: pytest.MonkeyPatch) -> None:
    default_rel = Relation(
        "tau_E_default",
        ("tau_E", "a"),
        symbol("tau_E") - 2.0 * symbol("a"),
        solve_for=("tau_E",),
    )
    override_rel = Relation(
        "tau_E_override",
        ("tau_E", "a"),
        symbol("tau_E") - 3.0 * symbol("a"),
        solve_for=("tau_E",),
    )
    monkeypatch.setattr(
        reactors_module.Reactor,
        "_RELATIONS",
        [(("plasma",), default_rel), (("plasma",), override_rel)],
        raising=False,
    )
    monkeypatch.setattr(reactors_module.Reactor, "_RELATIONS_IMPORTED", True, raising=False)

    reactor = Reactor(
        id="TEST",
        name="Test reactor",
        reactor_configuration="tokamak",
        organization="Example Lab",
        parameters={"a": 2.0},
        parameter_methods={"tau_E": "override"},
    )

    assert math.isclose(float(reactor.parameters["tau_E"]), 6.0)


def test_power_relations_computed_and_validated(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 3.0",
                "a: 1.0",
                "",
                "# plasma parameters",
                "B0: 5.0",
                "q95: 5.0",
                "",
                "# power and efficiency",
                "P_sep: 150e6",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.A), 3.0)
    assert math.isclose(float(reactor.P_sep_over_R), 50e6)
    assert math.isclose(
        float(reactor.P_sep_B_over_q95AR),
        150e6 * 5.0 / (5.0 * float(reactor.A) * float(reactor.R)),
    )


def test_inconsistent_power_relation_warns_and_updates_explicit(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 3.0",
                "a: 1.0",
                "",
                "# power and efficiency",
                "P_sep: 120e6",
                "P_sep_over_R: 30e6",  # should be 40e6
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.P_sep_over_R), 40e6)


def test_tokamak_volume_and_surface_backfilled(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "compact tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 4.0",
                "a: 1.2",
                "kappa: 1.6",
                "delta_95: 0.2",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    delta = 1.5 * 0.2
    expected_V = float(plasma_volume(1.2, 4.0, 1.6, delta, 0.0))
    expected_S = float(plasma_surface_area(1.2, 4.0, 1.6, delta, 0.0))

    assert math.isclose(float(reactor.V_p), expected_V, rel_tol=1e-6)
    assert math.isclose(float(reactor.S_p), expected_S, rel_tol=1e-6)


def test_tokamak_volume_inconsistency_warns_and_updates_explicit(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 4.0",
                "a: 1.2",
                "kappa: 1.6",
                "delta_95: 0.2",
                "V_p: 1.0",
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    expected_V = float(plasma_volume(1.2, 4.0, 1.6, 1.5 * 0.2, 0.0))
    assert math.isclose(float(reactor.V_p), expected_V, rel_tol=1e-6)


class TestDefaults:
    """Tests for default values set in reactor_defaults.py."""

    def test_T_e_from_T_avg(self, tmp_path: Path) -> None:
        """T_e should be set from T_avg when not provided."""
        reactor_dir = tmp_path / "reactors" / "T_E_TEST"
        reactor_dir.mkdir(parents=True)
        path = reactor_dir / "reactor.yaml"
        path.write_text(
            "\n".join(
                [
                    'id: "T_E_TEST"',
                    'name: "Temperature default test"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Test Lab"',
                    "",
                    "T_avg: 15.0",
                ]
            )
        )
        reactor = load_reactor_yaml(path)
        assert reactor.T_avg == 15.0
        assert reactor.T_e == 15.0

    def test_T_i_from_T_avg(self, tmp_path: Path) -> None:
        """T_i should be set from T_avg when not provided."""
        reactor_dir = tmp_path / "reactors" / "T_I_TEST"
        reactor_dir.mkdir(parents=True)
        path = reactor_dir / "reactor.yaml"
        path.write_text(
            "\n".join(
                [
                    'id: "T_I_TEST"',
                    'name: "Temperature default test"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Test Lab"',
                    "",
                    "T_avg: 12.0",
                ]
            )
        )
        reactor = load_reactor_yaml(path)
        assert reactor.T_avg == 12.0
        assert reactor.T_i == 12.0

    def test_n_la_from_n_avg(self, tmp_path: Path) -> None:
        """n_la should be set from n_avg when not provided."""
        reactor_dir = tmp_path / "reactors" / "N_LA_TEST"
        reactor_dir.mkdir(parents=True)
        path = reactor_dir / "reactor.yaml"
        path.write_text(
            "\n".join(
                [
                    'id: "N_LA_TEST"',
                    'name: "Density default test"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Test Lab"',
                    "",
                    "n_avg: 1.0e20",
                ]
            )
        )
        reactor = load_reactor_yaml(path)
        assert reactor.n_avg == 1.0e20
        assert reactor.n_la == 1.0e20

    def test_explicit_T_e_preserved(self, tmp_path: Path) -> None:
        """Explicit T_e should not be overwritten by default."""
        reactor_dir = tmp_path / "reactors" / "T_E_EXPLICIT_TEST"
        reactor_dir.mkdir(parents=True)
        path = reactor_dir / "reactor.yaml"
        path.write_text(
            "\n".join(
                [
                    'id: "T_E_EXPLICIT_TEST"',
                    'name: "Temperature explicit test"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Test Lab"',
                    "",
                    "T_avg: 15.0",
                    "T_e: 18.0",
                ]
            )
        )
        reactor = load_reactor_yaml(path)
        assert reactor.T_avg == 15.0
        assert reactor.T_e == 18.0
        assert reactor.T_i == 15.0

    def test_default_fractions(self, tmp_path: Path) -> None:
        """Default fractions should be 50-50 D-T."""
        reactor_dir = tmp_path / "reactors" / "FRAC_TEST"
        reactor_dir.mkdir(parents=True)
        path = reactor_dir / "reactor.yaml"
        path.write_text(
            "\n".join(
                [
                    'id: "FRAC_TEST"',
                    'name: "Fraction default test"',
                    'reactor_configuration: "tokamak"',
                    'organization: "Test Lab"',
                    "",
                    "R: 5.0",
                ]
            )
        )
        reactor = load_reactor_yaml(path)
        assert reactor.f_D == 0.5
        assert reactor.f_T == 0.5
        assert reactor.f_He3 == 0.0
        assert reactor.f_He4 == 0.0

