import math
from pathlib import Path

import pytest

from fusdb.geometry import plasma_surface_area, plasma_volume
from fusdb.loader import load_reactor_yaml


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
                    'allow_relation_overrides: true',
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
                    'allow_relation_overrides: true',
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


def test_parameters_and_artifacts_missing_defaults(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'allow_relation_overrides: true',
                    'organization: "Example Lab"',
                "",
                "plasma_parameters: {}",  # explicitly empty
                "",
                "power_and_efficiency: {}",  # explicitly empty
            ]
        )
    )

    reactor = load_reactor_yaml(path)
    assert reactor.P_fus is None
    assert reactor.tau_E is None
    assert reactor.has_density_profile() is False
    assert reactor.density_profile_file is None


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
                    'allow_relation_overrides: true',
                    'organization: "Example Lab"',
                "",
                "plasma_geometry:",
                "  R: 6.0",
                "  A: 3.0",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.a, 2.0)
    assert math.isclose(reactor.A, reactor.R / reactor.a)


def test_inconsistent_geometry_warns_and_sets_computed(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
                [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'allow_relation_overrides: true',
                    'organization: "Example Lab"',
                "",
                "plasma_geometry:",
                "  R: 3.0",
                "  a: 1.0",
                "  A: 2.5",
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.A, 3.0)
    assert math.isclose(reactor.R / reactor.a, reactor.A)


def test_geometry_within_tolerance_warns(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
                [
                    'id: "ARC_2018"',
                    'name: "ARC 2018 baseline"',
                    'reactor_configuration: "tokamak"',
                    'allow_relation_overrides: true',
                    'organization: "Example Lab"',
                "",
                "plasma_geometry:",
                "  R: 9.0",
                "  a: 2.9",
                "  A: 3.1",  # slightly off, but within 1%
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.A, 3.1)


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
                "plasma_geometry:",
                "  R_max: 5.0",
                "  R_min: 3.0",
                "  a: 1.0",
                "  Z_max: 2.0",
                "  Z_min: -2.0",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.R, 4.0)
    assert math.isclose(reactor.kappa, 2.0)


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
                "allow_relation_overrides: true",
                "",
                "plasma_geometry:",
                "  R: 3.0",
                "  a: 1.0",
                "",
                "plasma_parameters:",
                "  B0: 5.0",
                "  q95: 5.0",
                "",
                "power_and_efficiency:",
                "  P_sep: 150.0",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.A, 3.0)
    assert math.isclose(reactor.P_sep_over_R, 50.0)
    assert math.isclose(reactor.P_sep_B_over_q95AR, 150.0 * 5.0 / (5.0 * reactor.A * reactor.R))


def test_inconsistent_power_relation_warns_and_sets_computed(tmp_path: Path) -> None:
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
                "allow_relation_overrides: true",
                "",
                "plasma_geometry:",
                "  R: 3.0",
                "  a: 1.0",
                "",
                "power_and_efficiency:",
                "  P_sep: 120.0",
                "  P_sep_over_R: 30.0",  # should be 40
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.P_sep_over_R, 40.0)


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
                "plasma_geometry:",
                "  R: 4.0",
                "  a: 1.2",
                "  kappa: 1.6",
                "  delta_95: 0.2",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    expected_V = plasma_volume(1.2, 4.0, 1.6, 0.2, 0.0)
    expected_S = plasma_surface_area(1.2, 4.0, 1.6, 0.2, 0.0)

    assert math.isclose(reactor.V_p, expected_V, rel_tol=1e-6)
    assert math.isclose(reactor.S_p, expected_S, rel_tol=1e-6)


def test_tokamak_volume_inconsistency_warns(tmp_path: Path) -> None:
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
                "allow_relation_overrides: true",
                "",
                "plasma_geometry:",
                "  R: 4.0",
                "  a: 1.2",
                "  kappa: 1.6",
                "  delta_95: 0.2",
                "  V_p: 1.0",
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert math.isclose(reactor.V_p, plasma_volume(1.2, 4.0, 1.6, 0.2, 0.0))
