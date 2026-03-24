import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.registry import parse_variables
from fusdb.reactor_class import Reactor


def test_parse_variables_uses_density_reference_to_seed_species_profiles() -> None:
    """Expected: a fractions block seeds species density profiles when a density reference is provided."""
    parsed = parse_variables(
        {
            "f_BS": 0.63,
            "f_GW": {"value": 0.67, "fixed": True},
            "n_avg": 1.0e20,
            "fractions": {"D": 0.5, "T": 0.5},
        }
    )

    assert parsed["f_BS"].input_value == 0.63
    assert parsed["f_GW"].input_value == 0.67
    assert parsed["f_GW"].fixed is True
    assert np.mean(parsed["n_D"].input_value) == pytest.approx(5.0e19)
    assert np.mean(parsed["n_T"].input_value) == pytest.approx(5.0e19)
    assert "f_D" not in parsed or parsed["f_D"].input_value is None
    assert "f_T" not in parsed or parsed["f_T"].input_value is None


def test_arc_greenwald_fraction_is_loaded_and_kept_fixed() -> None:
    """Expected: ARC keeps its explicit fixed f_GW input while n_GW stays on the Greenwald limit relation."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")
    reactor.solve()

    assert reactor.variables_dict["f_BS"].input_value == 0.63
    assert reactor.variables_dict["f_GW"].input_value == 0.67
    assert reactor.variables_dict["f_GW"].current_value == 0.67
    assert reactor.variables_dict["f_GW"].fixed is True
    assert reactor.variables_dict["n_GW"].current_value == pytest.approx(
        1.0e20
        * (reactor.variables_dict["I_p"].current_value / 1.0e6)
        / (np.pi * reactor.variables_dict["a"].current_value ** 2),
        rel=1e-11,
    )


def test_arc_fraction_inputs_seed_density_profiles_and_derived_fractions() -> None:
    """Expected: ARC uses the fractions block as a density seed and derives integrated fractions from the solved densities."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")
    reactor.solve()

    assert reactor.variables_dict["f_D"].input_value is None
    assert reactor.variables_dict["f_T"].input_value is None
    assert np.mean(reactor.variables_dict["n_D"].input_value) == pytest.approx(0.5 * reactor.variables_dict["n_avg"].input_value)
    assert np.mean(reactor.variables_dict["n_T"].input_value) == pytest.approx(0.5 * reactor.variables_dict["n_avg"].input_value)
    assert reactor.variables_dict["f_D"].current_value < 0.5
    assert reactor.variables_dict["f_T"].current_value < 0.5
    assert reactor.variables_dict["f_He4"].current_value > 0.0
    assert (
        reactor.variables_dict["f_D"].current_value
        + reactor.variables_dict["f_T"].current_value
        + reactor.variables_dict["f_He3"].current_value
        + reactor.variables_dict["f_He4"].current_value
    ) == pytest.approx(1.0, abs=1e-12)
