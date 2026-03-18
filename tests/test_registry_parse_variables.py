import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.registry import parse_variables
from fusdb.reactor_class import Reactor


def test_parse_variables_preserves_non_species_fraction_inputs() -> None:
    """Expected: a fractions block only replaces species fractions and preserves unrelated f_* variables."""
    parsed = parse_variables(
        {
            "f_BS": 0.63,
            "f_GW": {"value": 0.67, "fixed": True},
            "fractions": {"D": 0.5, "T": 0.5},
        }
    )

    assert parsed["f_BS"].input_value == 0.63
    assert parsed["f_GW"].input_value == 0.67
    assert parsed["f_GW"].fixed is True
    assert parsed["f_D"].input_value == 0.5
    assert parsed["f_T"].input_value == 0.5


def test_arc_greenwald_fraction_is_loaded_and_kept_fixed() -> None:
    """Expected: ARC keeps its explicit fixed f_GW input and solves n_GW consistently from it."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")
    reactor.solve()

    assert reactor.variables_dict["f_BS"].input_value == 0.63
    assert reactor.variables_dict["f_GW"].input_value == 0.67
    assert reactor.variables_dict["f_GW"].current_value == 0.67
    assert reactor.variables_dict["f_GW"].fixed is True
    assert reactor.variables_dict["n_GW"].current_value == reactor.variables_dict["n_avg"].current_value / reactor.variables_dict["f_GW"].current_value


def test_arc_fraction_inputs_seed_steady_state_composition() -> None:
    """Expected: ARC keeps 50/50 D-T as the input seed, but the solved composition includes burn products."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")
    reactor.solve()

    assert reactor.variables_dict["f_D"].input_value == 0.5
    assert reactor.variables_dict["f_T"].input_value == 0.5
    assert reactor.variables_dict["f_D"].current_value < 0.5
    assert reactor.variables_dict["f_T"].current_value < 0.5
    assert reactor.variables_dict["f_He4"].current_value > 0.0
    assert (
        reactor.variables_dict["f_D"].current_value
        + reactor.variables_dict["f_T"].current_value
        + reactor.variables_dict["f_He3"].current_value
        + reactor.variables_dict["f_He4"].current_value
    ) == pytest.approx(1.0, abs=1e-12)
