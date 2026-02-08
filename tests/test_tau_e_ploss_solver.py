import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relations.confinement.scalings import tau_E_iter_ipb98y2
from fusdb.relations.confinement.plasma_stored_energy import energy_confinement_time
from fusdb.relationsystem_class import RelationSystem
from fusdb.reactor_class import Reactor
from fusdb.variable_class import Variable


def _make_input(name: str, value: float) -> Variable:
    var = Variable(name=name)
    var.values = [value]
    var.value_passes = [0]
    var.history = [{"pass_id": 0, "old": None, "new": value, "reason": "input"}]
    var.input_source = "explicit"
    return var


def _reference_tau_e_ploss(rel_scaling, w_th: float, inputs: dict[str, float]) -> tuple[float, float]:
    # Estimate power-law exponent from two points (cfspopcon-style)
    p1 = 1.0e6
    p2 = 1.0e7
    f1 = float(rel_scaling.evaluate(**{**inputs, "P_loss": p1}))
    f2 = float(rel_scaling.evaluate(**{**inputs, "P_loss": p2}))
    alpha = math.log(f2 / f1) / math.log(p2 / p1)
    gamma = f1 / (p1 ** alpha)
    p_loss = (w_th / gamma) ** (1.0 / (alpha + 1.0))
    tau_e = w_th / p_loss
    return tau_e, p_loss


def test_tau_e_ploss_matches_reference_ipb98y2():
    inputs = {
        "I_p": 15.0e6,
        "B0": 5.5,
        "n_la": 1.1e20,
        "R": 6.2,
        "kappa_ipb": 1.8,
        "A": 3.0,
        "afuel": 2.5,
    }
    w_th = 3.0e9

    tau_ref, p_loss_ref = _reference_tau_e_ploss(tau_E_iter_ipb98y2, w_th, inputs)

    variables = {name: _make_input(name, value) for name, value in inputs.items()}
    variables["W_th"] = _make_input("W_th", w_th)

    system = RelationSystem([tau_E_iter_ipb98y2, energy_confinement_time], variables, mode="overwrite")
    system.solve()

    tau_val = variables.get("tau_E").current_value if variables.get("tau_E") else None
    p_loss_val = variables.get("P_loss").current_value if variables.get("P_loss") else None

    assert tau_val is not None and p_loss_val is not None
    assert math.isfinite(float(tau_val))
    assert math.isfinite(float(p_loss_val))
    assert math.isclose(float(tau_val), tau_ref, rel_tol=1e-4)
    assert math.isclose(float(p_loss_val), p_loss_ref, rel_tol=1e-4)


def _get_float(variables: dict[str, Variable], name: str) -> float | None:
    value = Variable.get_from_dict(variables, name, allow_override=True, mode="current")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def test_reactor_yaml_against_cfspopcon_formula():
    reactor = Reactor.from_yaml("reactors/STEP_2024/reactor.yaml")
    reactor.solve(mode="overwrite")

    variables = reactor.variables_dict
    required = ["I_p", "B0", "n_la", "R", "kappa_ipb", "A", "afuel", "W_th"]
    values = {name: _get_float(variables, name) for name in required}
    if any(values[name] is None for name in required):
        import pytest
        missing = [name for name in required if values[name] is None]
        pytest.skip(f"Missing required reactor values: {missing}")

    inputs = {
        "I_p": values["I_p"],
        "B0": values["B0"],
        "n_la": values["n_la"],
        "R": values["R"],
        "kappa_ipb": values["kappa_ipb"],
        "A": values["A"],
        "afuel": values["afuel"],
    }
    w_th = values["W_th"]

    tau_ref, p_loss_ref = _reference_tau_e_ploss(tau_E_iter_ipb98y2, w_th, inputs)

    system_vars = {name: _make_input(name, value) for name, value in inputs.items()}
    system_vars["W_th"] = _make_input("W_th", w_th)
    system = RelationSystem([tau_E_iter_ipb98y2, energy_confinement_time], system_vars, mode="overwrite")
    system.solve()

    tau_val = system_vars.get("tau_E").current_value if system_vars.get("tau_E") else None
    p_loss_val = system_vars.get("P_loss").current_value if system_vars.get("P_loss") else None

    assert tau_val is not None and p_loss_val is not None
    assert math.isfinite(float(tau_val))
    assert math.isfinite(float(p_loss_val))
    assert math.isclose(float(tau_val), tau_ref, rel_tol=1e-4)
    assert math.isclose(float(p_loss_val), p_loss_ref, rel_tol=1e-4)
