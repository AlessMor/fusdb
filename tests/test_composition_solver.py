import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relations.plasma_composition.composition_solver import (
    CompositionSteadyStateSystem,
    solve_steady_state_composition,
)


def test_composition_rhs_preserves_tracked_fraction_total() -> None:
    """Expected: the dynamic RHS preserves the tracked-species total exactly."""
    system = CompositionSteadyStateSystem(
        n_i=1.0e20,
        T_avg=10.0,
        fractions={"f_D": 0.5, "f_T": 0.5, "f_imp": 0.1},
        tau_p={"tau_p": 1.0},
    )

    rhs = system.rhs(0.0, system.initial_state())
    assert float(np.sum(rhs)) == pytest.approx(0.0, abs=1e-12)


def test_zero_reactivity_solution_matches_source_distribution_for_equal_tau() -> None:
    """Expected: with zero burn and equal confinement times, the steady state matches the source mix."""
    fractions = solve_steady_state_composition(
        n_i=1.0e20,
        T_avg=0.0,
        fractions={"f_D": 0.5, "f_T": 0.5, "f_imp": 0.1},
        tau_p={"tau_p": 1.0},
    )

    assert fractions["f_D"] == pytest.approx(0.45, abs=1e-10)
    assert fractions["f_T"] == pytest.approx(0.45, abs=1e-10)
    assert fractions["f_He3"] == pytest.approx(0.0, abs=1e-10)
    assert fractions["f_He4"] == pytest.approx(0.0, abs=1e-10)
    assert sum(fractions.values()) == pytest.approx(0.9, abs=1e-10)


def test_initial_guess_does_not_change_zero_burn_steady_state() -> None:
    """Expected: the source mix is physical input, while initial_guess only seeds the root solve."""
    fractions = solve_steady_state_composition(
        n_i=1.0e20,
        T_avg=0.0,
        tau_p={"tau_p": 1.0},
        f_imp=0.1,
        source_distribution={"f_D": 0.2, "f_T": 0.8},
        initial_guess={"f_D": 0.5, "f_T": 0.5},
    )

    assert fractions["f_D"] == pytest.approx(0.18, abs=1e-10)
    assert fractions["f_T"] == pytest.approx(0.72, abs=1e-10)
    assert fractions["f_He3"] == pytest.approx(0.0, abs=1e-10)
    assert fractions["f_He4"] == pytest.approx(0.0, abs=1e-10)
    assert sum(fractions.values()) == pytest.approx(0.9, abs=1e-10)


def test_burning_solution_has_small_residual_and_positive_ash() -> None:
    """Expected: the root solve finds a non-trivial steady state with a small residual."""
    system = CompositionSteadyStateSystem(
        n_i=1.3e20,
        T_avg=14.0,
        fractions={"f_D": 0.5, "f_T": 0.5},
        tau_p={"tau_p_D": 1.0, "tau_p_T": 1.0, "tau_p_He3": 1.0, "tau_p_He4": 1.0},
    )

    fractions = system.solve()
    state = np.asarray(
        [fractions["f_D"], fractions["f_T"], fractions["f_He3"], fractions["f_He4"]],
        dtype=float,
    )
    residual = system.rhs(0.0, state)

    assert float(np.linalg.norm(residual[1:])) <= 1e-8
    assert fractions["f_He4"] > 0.0
    assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-10)
