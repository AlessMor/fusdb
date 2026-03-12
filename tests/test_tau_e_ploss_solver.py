import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusdb.relations.confinement.scalings import tau_E_iter_ipb98y2
from fusdb.relations.confinement.plasma_stored_energy import energy_confinement_time
from fusdb.relations.geometry.plasma_geometry import (
    sauter_plasma_cross_sectional_surface,
    sauter_plasma_volume,
    kappa_ipb_from_volume,
)
from fusdb.relationsystem_class import RelationSystem
from fusdb.reactor_class import Reactor
from fusdb.registry import KEV_TO_J
from fusdb.variable_class import Variable
from fusdb.variable_util import make_variable


def _make_input(name: str, value: float) -> Variable:
    var = make_variable(name=name, ndim=0)
    var.add_value(value, as_input=True)
    var.input_source = "explicit"
    return var


def _reference_tau_e_ploss(rel_scaling, w_th: float, inputs: dict[str, float]) -> tuple[float, float]:
    # Estimate power-law exponent from two points (cfspopcon-style)
    p1 = 1.0e6
    p2 = 1.0e7
    f1 = float(rel_scaling.evaluate({**inputs, "P_loss": p1}))
    f2 = float(rel_scaling.evaluate({**inputs, "P_loss": p2}))
    alpha = math.log(f2 / f1) / math.log(p2 / p1)
    gamma = f1 / (p1 ** alpha)
    p_loss = (w_th / gamma) ** (1.0 / (alpha + 1.0))
    tau_e = w_th / p_loss
    return tau_e, p_loss


def test_tau_e_ploss_matches_reference_ipb98y2():
    """Expected: coupled tau_E/P_loss solve matches the reference power-law fixed-point solution."""
    inputs = {
        "H98_y2": 1.0,
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

    variables_list = [_make_input(name, value) for name, value in inputs.items()]
    variables_list.append(_make_input("W_th", w_th))

    system = RelationSystem([tau_E_iter_ipb98y2, energy_confinement_time], variables_list, mode="overwrite")
    system.solve()

    tau_var = system._graph["vars"].get("tau_E")
    p_loss_var = system._graph["vars"].get("P_loss")

    tau_val = float(tau_var.current_value) if tau_var and tau_var.current_value is not None else None
    p_loss_val = float(p_loss_var.current_value) if p_loss_var and p_loss_var.current_value is not None else None

    assert tau_val is not None and p_loss_val is not None
    assert math.isfinite(tau_val)
    assert math.isfinite(p_loss_val)
    assert math.isclose(tau_val, tau_ref, rel_tol=1e-4), f"tau_E: {tau_val} vs ref {tau_ref}"
    assert math.isclose(p_loss_val, p_loss_ref, rel_tol=1e-4), f"P_loss: {p_loss_val} vs ref {p_loss_ref}"


def _get_float(variables: dict[str, Variable], name: str) -> float | None:
    var = variables.get(name)
    value = None if var is None else var.current_value
    if value is None and var is not None:
        value = var.input_value
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _manual_ipb98_solution_from_reactor_input(reactor: Reactor) -> tuple[float, float, float]:
    def get_input(name: str) -> float | None:
        var = reactor.variables_dict.get(name)
        return None if var is None else var.input_value

    def require(name: str, value: float | None) -> float:
        if value is None:
            raise ValueError(f"Missing required input '{name}'")
        return value

    R = require("R", get_input("R"))
    a = require("a", get_input("a"))
    A = get_input("A") or R / a
    I_p = require("I_p", get_input("I_p"))
    B0 = require("B0", get_input("B0"))
    H98_y2 = require("H98_y2", get_input("H98_y2"))
    afuel = require("afuel", get_input("afuel"))

    n_avg = get_input("n_avg")
    f_GW = get_input("f_GW")
    if n_avg is None and f_GW is not None:
        n_GW = 1e20 * (I_p / 1e6) / (math.pi * a**2)
        n_avg = f_GW * n_GW
    n_avg = require("n_avg (or f_GW)", n_avg)

    n_la = get_input("n_la") or n_avg
    n_la = require("n_la", n_la)

    T_avg = get_input("T_avg")
    T_e = get_input("T_e") or T_avg
    T_i = get_input("T_i") or T_avg
    T_e = require("T_e (or T_avg)", T_e)
    T_i = require("T_i (or T_avg)", T_i)

    n_e = get_input("n_e") or n_avg
    n_i = get_input("n_i") or n_avg

    V_p = get_input("V_p")
    kappa_ipb = get_input("kappa_ipb")
    if V_p is None or kappa_ipb is None:
        kappa = get_input("kappa")
        kappa_95 = get_input("kappa_95")
        if kappa is None and kappa_95 is not None:
            kappa = 1.12 * kappa_95

        delta = get_input("delta")
        delta_95 = get_input("delta_95")
        if delta is None and delta_95 is not None:
            delta = 1.5 * delta_95

        squareness = get_input("squareness") or 0.0

        if V_p is None:
            if kappa is None or delta is None:
                raise ValueError("Need V_p or (kappa/kappa_95 and delta/delta_95) to compute V_p.")
            eps = a / R
            S_phi = float(
                sauter_plasma_cross_sectional_surface.evaluate(
                    {"a": a, "kappa": kappa, "delta": delta, "squareness": squareness}
                )
            )
            V_p = float(
                sauter_plasma_volume.evaluate(
                    {"R": R, "delta": delta, "eps": eps, "S_phi": S_phi}
                )
            )

        if kappa_ipb is None:
            kappa_ipb = float(
                kappa_ipb_from_volume.evaluate(
                    {"V_p": V_p, "R": R, "a": a}
                )
            )

    p_th = (n_e * T_e + n_i * T_i) * KEV_TO_J
    W_th = 1.5 * p_th * V_p

    alpha_P = -0.69
    gamma = (
        H98_y2
        * 0.0562
        * (I_p / 1e6) ** 0.93
        * B0 ** 0.15
        * (n_la / 1e19) ** 0.41
        * R ** 1.97
        * kappa_ipb ** 0.78
        * A ** (-0.58)
        * afuel ** 0.19
        * (1e6) ** 0.69
    )

    if gamma > 0.0:
        P_loss = (W_th / gamma) ** (1.0 / (1.0 + alpha_P))
    else:
        P_loss = math.inf

    tau_E = W_th / P_loss
    return tau_E, P_loss, W_th


def test_reactor_yaml_against_cfspopcon_formula():
    """Expected: DEMO_2022 solved tau_E, P_loss, and W_th match manual CFSPopCon-style calculations."""
    reactor_path = "reactors/DEMO_2022/reactor.yaml"
    manual_reactor = Reactor.from_yaml(reactor_path)
    tau_manual, p_loss_manual, w_th_manual = _manual_ipb98_solution_from_reactor_input(manual_reactor)

    reactor = Reactor.from_yaml(reactor_path)
    reactor.solve(mode="overwrite")
    variables = reactor.variables_dict

    tau_overwrite = _get_float(variables, "tau_E")
    p_loss_overwrite = _get_float(variables, "P_loss")
    w_th_overwrite = _get_float(variables, "W_th")

    assert tau_overwrite is not None and p_loss_overwrite is not None and w_th_overwrite is not None
    assert math.isfinite(float(tau_overwrite))
    assert math.isfinite(float(p_loss_overwrite))
    assert math.isfinite(float(w_th_overwrite))

    assert math.isclose(float(p_loss_overwrite), p_loss_manual, rel_tol=1e-3)
    assert math.isclose(float(tau_overwrite), tau_manual, rel_tol=1e-3)
    assert math.isclose(float(w_th_overwrite), w_th_manual, rel_tol=1e-3)
