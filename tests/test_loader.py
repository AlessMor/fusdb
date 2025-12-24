import math
from pathlib import Path
import pytest

from fusiondb.loader import find_reactor_dirs, load_all_reactors, load_reactor_yaml


def _write_sample_reactor(tmp_path: Path) -> Path:
    reactor_dir = tmp_path / "reactors" / "ARC_2018"
    reactor_dir.mkdir(parents=True)
    reactor_yaml = reactor_dir / "reactor.yaml"
    reactor_yaml.write_text(
        "\n".join(
            [
                'id: "ARC_2018"',
                'reactor_class: "ARC"',
                'name: "ARC 2018 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "plasma_geometry:",
                "  R: 3.3",
                "  a: 1.13",
                "",
                "plasma_parameters:",
                "  B0: 5.0",
                "  n_avg: 1.1e20",
                "",
                "power_and_efficiency:",
                "  P_fus: 525",
                "",
                "artifacts:",
                "  density_profile:",
                '    file: "density_profile.h5"',
                '    x_axis: "rho"',
                '    y_dataset: "ne"',
            ]
        )
    )
    (reactor_dir / "density_profile.h5").touch()
    return reactor_yaml


def test_load_reactor_yaml(tmp_path: Path) -> None:
    yaml_path = _write_sample_reactor(tmp_path)

    reactor = load_reactor_yaml(yaml_path)

    assert reactor.id == "ARC_2018"
    assert reactor.name == "ARC 2018 baseline"
    assert reactor.reactor_configuration == "tokamak"
    assert reactor.organization == "Example Lab"
    assert reactor.P_fus == 525
    assert reactor.R == 3.3
    assert reactor.a == 1.13
    assert math.isclose(reactor.A, reactor.R / reactor.a)
    assert reactor.B0 == 5.0
    assert reactor.n_avg == "1.1e20"
    assert reactor.density_profile_file == "density_profile.h5"
    assert reactor.density_profile_x_axis == "rho"
    assert reactor.density_profile_y_dataset == "ne"
    assert reactor.root_dir == yaml_path.parent


def test_find_reactor_dirs_and_load_all(tmp_path: Path) -> None:
    yaml_path = _write_sample_reactor(tmp_path)
    reactor_dirs = find_reactor_dirs(tmp_path)
    assert reactor_dirs == [yaml_path.parent]

    reactors = load_all_reactors(tmp_path)
    assert set(reactors.keys()) == {"ARC_2018"}
    assert reactors["ARC_2018"].P_fus == 525


def test_missing_density_profile_file_warns(tmp_path: Path) -> None:
    reactor_dir = tmp_path / "reactors" / "ARC_2019"
    reactor_dir.mkdir(parents=True)
    path = reactor_dir / "reactor.yaml"
    path.write_text(
        "\n".join(
            [
                'id: "ARC_2019"',
                'reactor_class: "ARC"',
                'name: "ARC 2019 baseline"',
                'reactor_configuration: "tokamak"',
                'organization: "Example Lab"',
                "",
                "plasma_geometry:",
                "  R: 3.3",
                "  a: 1.13",
                "",
                "power_and_efficiency:",
                "  P_fus: 525",
                "",
                "artifacts:",
                "  density_profile:",
                '    file: "missing_profile.h5"',
                '    x_axis: "rho"',
                '    y_dataset: "ne"',
            ]
        )
    )

    with pytest.warns(UserWarning):
        reactor = load_reactor_yaml(path)

    assert reactor.density_profile_file == "missing_profile.h5"
