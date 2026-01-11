import math
import warnings
from pathlib import Path
import pytest

from fusdb.loader import find_reactor_dirs, load_all_reactors, load_reactor_yaml


def _write_sample_reactor(tmp_path: Path) -> Path:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    reactor_yaml = reactor_dir / "reactor.yaml"
    reactor_yaml.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'reactor_family: "ARC"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R: 3.3",
                "a: 1.13",
                "",
                "# plasma parameters",
                "B0: 5.0",
                "n_avg: 1.1e20",
                "",
                "# power and efficiency",
                "P_fus: 525e6",
            ]
        )
    )
    return reactor_yaml


def _write_country_reactor(tmp_path: Path, country: str) -> Path:
    reactor_dir = tmp_path / "reactors" / "COUNTRY_TEST"
    reactor_dir.mkdir(parents=True)
    reactor_yaml = reactor_dir / "reactor.yaml"
    reactor_yaml.write_text(
        "\n".join(
            [
                'id: "COUNTRY_TEST"',
                'name: "Country Test"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                f"country: {country}",
            ]
        )
    )
    return reactor_yaml


def test_load_reactor_yaml(tmp_path: Path) -> None:
    yaml_path = _write_sample_reactor(tmp_path)

    reactor = load_reactor_yaml(yaml_path)

    assert reactor.id == "ARC_2018"
    assert reactor.name == "ARC 2018 baseline"
    assert reactor.reactor_configuration == "tokamak"
    assert reactor.organization == "Example Lab"
    assert float(reactor.P_fus) == 525e6
    assert math.isclose(float(reactor.R), 3.3)
    assert math.isclose(float(reactor.a), 1.13)
    assert math.isclose(float(reactor.A), float(reactor.R) / float(reactor.a))
    assert math.isclose(float(reactor.B0), 5.0)
    assert math.isclose(float(reactor.n_avg), 1.1e20)
    assert reactor.root_dir == yaml_path.parent


def test_country_alpha3_preserved(tmp_path: Path) -> None:
    yaml_path = _write_country_reactor(tmp_path, "USA")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reactor = load_reactor_yaml(yaml_path)

    assert reactor.country == "USA"
    assert not any("Normalized country" in str(w.message) for w in caught)


def test_country_alpha2_normalized(tmp_path: Path) -> None:
    yaml_path = _write_country_reactor(tmp_path, "US")

    with pytest.warns(UserWarning, match="Normalized country"):
        reactor = load_reactor_yaml(yaml_path)

    assert reactor.country == "USA"


def test_country_name_normalized(tmp_path: Path) -> None:
    yaml_path = _write_country_reactor(tmp_path, "United States")

    with pytest.warns(UserWarning, match="Normalized country"):
        reactor = load_reactor_yaml(yaml_path)

    assert reactor.country == "USA"


def test_country_unknown_raises(tmp_path: Path) -> None:
    yaml_path = _write_country_reactor(tmp_path, "Atlantis")

    with pytest.raises(ValueError, match="country must be ISO"):
        load_reactor_yaml(yaml_path)


def test_find_reactor_dirs_and_load_all(tmp_path: Path) -> None:
    yaml_path = _write_sample_reactor(tmp_path)
    reactor_dirs = find_reactor_dirs(tmp_path)
    assert reactor_dirs == [yaml_path.parent]

    reactors = load_all_reactors(tmp_path)
    assert set(reactors.keys()) == {"ARC_2018"}
    assert float(reactors["ARC_2018"].P_fus) == 525e6


def test_parameter_tolerance_parsing(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2020"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2020"',
                'name: "ARC 2020 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "# plasma geometry",
                "R:",
                "  value: 3.3",
                "  tol: 0.05",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.R), 3.3)
    assert math.isclose(reactor.parameter_tolerances["R"], 0.05)


def test_parameter_method_parsing(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2021"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2021"',
                'name: "ARC 2021 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "custom_param:",
                "  method: IPB98",
                "tau_E:",
                "  value: 3.2",
                "  method: tau_E_iter_ipb98y2",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert reactor.parameters.get("custom_param") is None
    assert reactor.parameter_methods["custom_param"] == "IPB98"
    assert reactor.parameter_methods["tau_E"] == "tau_E_iter_ipb98y2"


def test_parameter_alias_mapping(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2022"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2022"',
                'name: "ARC 2022 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "elongation: 1.7",
                "triangularity:",
                "  value: 0.3",
                "  tol: 0.05",
                "xi: 0.9",
            ]
        )
    )

    reactor = load_reactor_yaml(path)

    assert math.isclose(float(reactor.parameters["kappa"]), 1.7)
    assert math.isclose(float(reactor.parameters["delta"]), 0.3)
    assert math.isclose(reactor.parameter_tolerances["delta"], 0.05)
    assert math.isclose(float(reactor.parameters["squareness"]), 0.9)


def test_duplicate_alias_raises(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2023"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2023"',
                'name: "ARC 2023 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "kappa: 1.8",
                "elongation: 1.9",
            ]
        )
    )

    with pytest.raises(ValueError, match="Duplicate parameter 'kappa'"):
        load_reactor_yaml(path)
