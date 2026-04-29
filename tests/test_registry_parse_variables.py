import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fusdb.registry.reactor_defaults as reactor_defaults
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
    """Expected: ARC keeps its explicit fixed f_GW input from YAML parsing."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")

    assert reactor.variables_dict["f_BS"].input_value == 0.63
    assert reactor.variables_dict["f_GW"].input_value == 0.67
    assert reactor.variables_dict["f_GW"].fixed is True


def test_arc_fraction_inputs_seed_density_profiles_and_derived_fractions() -> None:
    """Expected: ARC uses the fractions block as a density seed at load time."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "ARC_2015" / "reactor.yaml")

    assert "f_D" not in reactor.variables_dict or reactor.variables_dict["f_D"].input_value is None
    assert "f_T" not in reactor.variables_dict or reactor.variables_dict["f_T"].input_value is None
    assert np.mean(reactor.variables_dict["n_D"].input_value) == pytest.approx(0.5 * reactor.variables_dict["n_avg"].input_value)
    assert np.mean(reactor.variables_dict["n_T"].input_value) == pytest.approx(0.5 * reactor.variables_dict["n_avg"].input_value)


def test_stellaris_defaults_seed_species_densities_from_n_avg() -> None:
    """Expected: species defaults are seeded from n_avg before solving without fake relation inputs."""
    reactor = Reactor.from_yaml(ROOT / "reactors" / "STELLARIS" / "reactor.yaml")

    n_avg = float(reactor.variables_dict["n_avg"].input_value)
    n_d = reactor.variables_dict["n_D"]
    n_t = reactor.variables_dict["n_T"]

    assert n_d.input_value is not None
    assert n_t.input_value is not None
    assert np.mean(n_d.input_value) == pytest.approx(0.5 * n_avg)
    assert np.mean(n_t.input_value) == pytest.approx(0.5 * n_avg)

    for rel in reactor.default_relations:
        assert all(not name.startswith("_") for name in rel.inputs)


def test_parse_variables_rejects_abs_tol_key() -> None:
    """Expected: variable-level abs_tol input is unsupported and rejected."""
    with pytest.raises(ValueError, match="unsupported key 'abs_tol'"):
        parse_variables({"R": {"value": 3.0, "abs_tol": 1e-6}})


def test_registry_declared_profile_variable_keeps_array_runtime_shape() -> None:
    """Expected: ndim=1 registry variables convert scalar convenience input to flat arrays immediately."""
    parsed = parse_variables({"n_i": 1.2e20})
    var = parsed["n_i"]

    assert var.ndim == 1
    assert isinstance(var.input_value, np.ndarray)
    assert isinstance(var.current_value, np.ndarray)
    assert var.input_value.ndim == 1
    assert var.current_value.ndim == 1
    assert np.allclose(var.input_value, np.full(var.profile_size, 1.2e20))


def test_reactor_from_yaml_rejects_solver_abs_tol(tmp_path) -> None:
    """Expected: solver_tags.abs_tol is unsupported and rejected at load time."""
    reactor_yaml = tmp_path / "reactor.yaml"
    reactor_yaml.write_text(
        "\n".join(
            (
                "metadata:",
                "  id: R_abs_tol_reject",
                "solver_tags:",
                "  mode: overwrite",
                "  abs_tol: 1e-6",
                "variables:",
                "  R:",
                "    value: 3.0",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="solver_tags\\.abs_tol"):
        Reactor.from_yaml(reactor_yaml)


def test_reactor_from_yaml_propagates_defaults_errors(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Expected: defaults application failures propagate and fail reactor loading."""
    reactor_yaml = tmp_path / "reactor.yaml"
    reactor_yaml.write_text(
        "\n".join(
            (
                "metadata:",
                "  id: R_defaults_failure",
                "variables:",
                "  R:",
                "    value: 3.0",
            )
        ),
        encoding="utf-8",
    )

    def _raise_defaults_error(*_args, **_kwargs):
        raise RuntimeError("defaults exploded")

    monkeypatch.setattr(reactor_defaults, "apply_reactor_defaults", _raise_defaults_error)
    with pytest.raises(RuntimeError, match="defaults exploded"):
        Reactor.from_yaml(reactor_yaml)
