import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relations.plasma_composition.composition_solver import steady_state_plasma_composition
from fusdb.relations.reactivities.reactivity_functions import sigmav_DT_BoschHale
from fusdb.relationsystem_class import RelationSystem
from fusdb.variable_util import make_variable


_DENSITY_INPUTS = tuple(name for name in steady_state_plasma_composition.inputs if name.startswith("n_"))
_REACTIVITY_INPUTS = tuple(name for name in steady_state_plasma_composition.inputs if name.startswith("sigmav_"))
_TAU_INPUTS = tuple(name for name in steady_state_plasma_composition.inputs if name.startswith("tau_p_"))


def _equal_tau_inputs(value: float = 1.0) -> dict[str, float]:
    """Return equal confinement times for every tracked species."""
    return {name: value for name in _TAU_INPUTS}


def _uniform_density_inputs(density_D: float, density_T: float, density_He3: float, density_He4: float, *, n: int = 11) -> dict[str, np.ndarray]:
    """Return uniform tracked density profiles."""
    return {
        "n_D": np.full(n, density_D, dtype=float),
        "n_T": np.full(n, density_T, dtype=float),
        "n_He3": np.full(n, density_He3, dtype=float),
        "n_He4": np.full(n, density_He4, dtype=float),
    }


def _uniform_reactivity_inputs(*, sigmav_DT: float = 0.0, n: int = 11) -> dict[str, np.ndarray]:
    """Return uniform reactivity profiles."""
    return {
        "sigmav_DT": np.full(n, sigmav_DT, dtype=float),
        **{
            name: np.zeros(n, dtype=float)
            for name in _REACTIVITY_INPUTS
            if name != "sigmav_DT"
        },
    }


def test_zero_reactivity_solution_matches_seed_densities_for_equal_tau() -> None:
    """Expected: with zero burn and equal confinement times, the steady state matches the seeded density mix."""
    solved = steady_state_plasma_composition(
        **_uniform_density_inputs(4.5e19, 4.5e19, 0.0, 0.0),
        **_uniform_reactivity_inputs(),
        **_equal_tau_inputs(),
    )

    assert np.allclose(solved["n_D"], np.full(11, 4.5e19, dtype=float), rtol=1e-12, atol=0.0)
    assert np.allclose(solved["n_T"], np.full(11, 4.5e19, dtype=float), rtol=1e-12, atol=0.0)
    assert np.allclose(solved["n_He3"], np.zeros(11, dtype=float), rtol=0.0, atol=1e-12)
    assert np.allclose(solved["n_He4"], np.zeros(11, dtype=float), rtol=0.0, atol=1e-12)
    assert np.allclose(
        solved["n_D"] + solved["n_T"] + solved["n_He3"] + solved["n_He4"],
        np.full(11, 9.0e19, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_direct_call_rejects_non_profile_density_and_reactivity_inputs() -> None:
    """Expected: direct relation calls require 1D profiles for every density and reactivity input."""
    with pytest.raises(TypeError, match="Density and reactivity inputs must already be 1D numpy arrays"):
        steady_state_plasma_composition(
            n_D=4.5e19,
            n_T=4.5e19,
            n_He3=0.0,
            n_He4=0.0,
            sigmav_DT=0.0,
            sigmav_DDn=0.0,
            sigmav_DDp=0.0,
            sigmav_DHe3=0.0,
            sigmav_TT=0.0,
            sigmav_He3He3=0.0,
            sigmav_THe3_D=0.0,
            sigmav_THe3_np=0.0,
            **_equal_tau_inputs(),
        )


def test_burning_solution_has_positive_ash_and_preserves_total_density() -> None:
    """Expected: the direct solve finds a non-trivial steady state with positive alpha ash."""
    solved = steady_state_plasma_composition(
        **_uniform_density_inputs(6.5e19, 6.5e19, 0.0, 0.0),
        **_uniform_reactivity_inputs(sigmav_DT=float(sigmav_DT_BoschHale(14.0))),
        **_equal_tau_inputs(),
    )

    assert np.all(solved["n_D"] < 6.5e19)
    assert np.all(solved["n_T"] < 6.5e19)
    assert np.all(solved["n_He4"] > 0.0)
    assert np.allclose(
        solved["n_D"] + solved["n_T"] + solved["n_He3"] + solved["n_He4"],
        np.full(11, 1.3e20, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_composition_relation_updates_density_bundle() -> None:
    """Expected: the composition relation overwrites the seeded density bundle with one atomic solve."""
    variables = []
    inputs = {
        "n_D": 6.5e19,
        "n_T": 6.5e19,
        "n_He3": 0.0,
        "n_He4": 0.0,
        "sigmav_DT": float(sigmav_DT_BoschHale(14.0)),
        **{name: 0.0 for name in _REACTIVITY_INPUTS if name != "sigmav_DT"},
        **_equal_tau_inputs(),
    }
    for name, value in inputs.items():
        var = make_variable(name=name, ndim=1 if name in _DENSITY_INPUTS or name in _REACTIVITY_INPUTS else 0)
        var.add_value(value, as_input=True)
        variables.append(var)

    system = RelationSystem([steady_state_plasma_composition], variables, mode="overwrite")
    system.solve()
    solved = {
        name: np.asarray(system.variables_dict[name].current_value, dtype=float)
        for name in _DENSITY_INPUTS
    }

    assert float(np.mean(solved["n_He4"])) > 0.0
    assert float(np.mean(sum(solved.values()))) == pytest.approx(1.3e20, rel=1e-12)


def test_composition_relation_preserves_profile_shape() -> None:
    """Expected: profile inputs stay profile-shaped and are solved pointwise."""
    n_d = np.full(11, 6.5e19, dtype=float)
    n_t = np.full(11, 6.5e19, dtype=float)
    n_he3 = np.zeros(11, dtype=float)
    n_he4 = np.zeros(11, dtype=float)
    profile_inputs = {
        "n_D": n_d,
        "n_T": n_t,
        "n_He3": n_he3,
        "n_He4": n_he4,
        **_uniform_reactivity_inputs(sigmav_DT=float(sigmav_DT_BoschHale(14.0)), n=11),
        **_equal_tau_inputs(),
    }

    variables = []
    for name, value in profile_inputs.items():
        var = make_variable(name=name, ndim=1 if name in _DENSITY_INPUTS or name in _REACTIVITY_INPUTS else 0)
        var.add_value(value, as_input=True)
        variables.append(var)

    system = RelationSystem([steady_state_plasma_composition], variables, mode="overwrite")
    system.solve()

    solved = {
        name: np.asarray(system.variables_dict[name].current_value, dtype=float)
        for name in _DENSITY_INPUTS
    }
    solved_total = solved["n_D"] + solved["n_T"] + solved["n_He3"] + solved["n_He4"]

    assert solved["n_He4"].shape == n_d.shape
    assert np.all(solved["n_He4"] > 0.0)
    assert np.allclose(solved_total, np.full_like(n_d, 1.3e20), rtol=1e-12, atol=0.0)


def test_relationsystem_evaluate_rejects_non_array_profile_inputs() -> None:
    """Expected: evaluate() rejects scalar stand-ins for ndim=1 profile inputs."""
    variables = []
    seen = set()
    for name in steady_state_plasma_composition.inputs:
        var = make_variable(name=name, ndim=1 if name in _DENSITY_INPUTS or name in _REACTIVITY_INPUTS else 0)
        variables.append(var)
        seen.add(var.name)
    for name in steady_state_plasma_composition.outputs:
        if name not in seen:
            variables.append(make_variable(name=name, ndim=1))
            seen.add(name)

    system = RelationSystem([steady_state_plasma_composition], variables, mode="overwrite")
    with pytest.raises(TypeError, match="must already be provided as a numpy.ndarray"):
        system.evaluate(
            {
                "n_D": 6.5e19,
                "n_T": 6.5e19,
                "n_He3": 0.0,
                "n_He4": 0.0,
                "sigmav_DT": float(sigmav_DT_BoschHale(14.0)),
                **{name: 0.0 for name in _REACTIVITY_INPUTS if name != "sigmav_DT"},
                **_equal_tau_inputs(),
            }
        )


def test_relationsystem_evaluate_accepts_profile_arrays() -> None:
    """Expected: evaluate() still works when every ndim=1 input is already an array."""
    variables = []
    seen = set()
    for name in steady_state_plasma_composition.inputs:
        var = make_variable(name=name, ndim=1 if name in _DENSITY_INPUTS or name in _REACTIVITY_INPUTS else 0)
        variables.append(var)
        seen.add(var.name)
    for name in steady_state_plasma_composition.outputs:
        if name not in seen:
            variables.append(make_variable(name=name, ndim=1))
            seen.add(name)

    system = RelationSystem([steady_state_plasma_composition], variables, mode="overwrite")
    values = system.evaluate(
        {
            **_uniform_density_inputs(6.5e19, 6.5e19, 0.0, 0.0, n=51),
            **_uniform_reactivity_inputs(sigmav_DT=float(sigmav_DT_BoschHale(14.0)), n=51),
            **_equal_tau_inputs(),
        }
    )

    he4 = np.asarray(values["n_He4"], dtype=float)
    assert he4.shape == (51,)
    assert np.all(he4 > 0.0)
