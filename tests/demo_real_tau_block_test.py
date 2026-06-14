"""Standalone DEMO tau_E ordered-block vs reconcile test.

Reads the uploaded DEMO reactor.yaml, computes upstream quantities with the
same formulas used in the FusDB relations discussed in chat, then compares:
  1. ordered without simultaneous block
  2. ordered with [IPB98 scaling, energy confinement balance]
  3. full simultaneous reconcile of the same equations
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares

KEV_TO_J = 1.602176634e-16


def _as_float(value: Any) -> float:
    if isinstance(value, str):
        return float(value)
    return float(value)


def load_vars(path: Path) -> dict[str, float]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, float] = {}
    for name, item in raw["variables"].items():
        if isinstance(item, dict):
            value = _as_float(item.get("value"))
            unit = item.get("unit")
        else:
            value = _as_float(item)
            unit = None
        if unit == "MW":
            value *= 1.0e6
        elif unit == "GW":
            value *= 1.0e9
        elif unit == "MW/m":
            value *= 1.0e6
        elif unit == "MW*T/m":
            value *= 1.0e6
        out[name] = value
    return out


def sauter_cross_section(a: float, kappa: float, delta: float, squareness: float) -> float:
    theta_07 = math.asin(0.7) - 2.0 * squareness / (1.0 + math.sqrt(1.0 + 8.0 * squareness**2))
    w_07 = (
        math.cos(theta_07 - squareness * math.sin(2.0 * theta_07))
        / math.sqrt(0.51)
        * (1.0 - 0.49 / 2.0 * delta**2)
    )
    return math.pi * a**2 * kappa * (1.0 + 0.52 * (w_07 - 1.0))


def sauter_volume(R: float, delta: float, eps: float, S_phi: float) -> float:
    return 2.0 * math.pi * R * (1.0 - 0.25 * delta * eps) * S_phi


def greenwald_density_limit(I_p: float, a: float) -> float:
    return 1.0e20 * (I_p / 1.0e6) / (math.pi * a**2)


def thermal_pressure_uniform(n_avg: float, T_avg: float) -> float:
    # Same result as trapz over rho in [0, 1] for uniform n_i=n_e=n_avg and T_i=T_e=T_avg.
    return 2.0 * n_avg * T_avg * KEV_TO_J


def thermal_stored_energy(p_th: float, V_p: float) -> float:
    return 1.5 * p_th * V_p


def tau_ipb98_iter_user(
    I_p: float,
    B0: float,
    n_la: float,
    P_loss: float,
    R: float,
    kappa: float,
    A: float,
    afuel: float,
    H98_y2: float,
) -> float:
    dnla19 = n_la / 1.0e19
    I_p_MA = I_p / 1.0e6
    P_loss_MW = P_loss / 1.0e6
    return H98_y2 * (
        0.0365
        * I_p_MA**0.97
        * B0**0.08
        * dnla19**0.41
        * P_loss_MW ** (-0.63)
        * R**1.93
        * kappa**0.67
        * A ** (-0.23)
        * afuel**0.2
    )


def residual_scale(value: float) -> float:
    return max(abs(float(value)), 1.0)


def solve_tau_block(constants: dict[str, float], W_th: float) -> dict[str, float]:
    def residual(y: np.ndarray) -> np.ndarray:
        log_P_loss, log_tau_E = y
        P_loss = math.exp(log_P_loss)
        tau_E = math.exp(log_tau_E)
        tau_model = tau_ipb98_iter_user(P_loss=P_loss, **{k: v for k, v in constants.items() if k != "P_aux_seed"})
        return np.array([
            (tau_E - tau_model) / residual_scale(tau_model),
            (P_loss * tau_E - W_th) / residual_scale(W_th),
        ])

    x0 = np.log([constants.get("P_aux_seed", 200.0e6), 4.0])
    sol = least_squares(residual, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    P_loss, tau_E = np.exp(sol.x)
    return {
        "success": bool(sol.success and np.linalg.norm(residual(sol.x), ord=np.inf) < 1e-8),
        "P_loss": float(P_loss),
        "tau_E": float(tau_E),
        "max_residual": float(np.linalg.norm(residual(sol.x), ord=np.inf)),
        "nfev": int(sol.nfev),
        "message": str(sol.message),
    }


def solve_reconcile(constants: dict[str, float], upstream: dict[str, float]) -> dict[str, float]:
    target_V = upstream["V_p"]
    target_p = upstream["p_th"]
    # Unknowns are log(V_p), log(p_th), log(W_th), log(P_loss), log(tau_E)
    def residual(y: np.ndarray) -> np.ndarray:
        V_p, p_th, W_th, P_loss, tau_E = np.exp(y)
        tau_model = tau_ipb98_iter_user(P_loss=P_loss, **{k: v for k, v in constants.items() if k != "P_aux_seed"})
        return np.array([
            (V_p - target_V) / residual_scale(target_V),
            (p_th - target_p) / residual_scale(target_p),
            (W_th - thermal_stored_energy(p_th, V_p)) / residual_scale(W_th),
            (tau_E - tau_model) / residual_scale(tau_model),
            (P_loss * tau_E - W_th) / residual_scale(W_th),
        ])

    x0 = np.log([upstream["V_p"], upstream["p_th"], upstream["W_th"], constants.get("P_aux_seed", 200.0e6), 4.0])
    sol = least_squares(residual, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    V_p, p_th, W_th, P_loss, tau_E = np.exp(sol.x)
    return {
        "success": bool(sol.success and np.linalg.norm(residual(sol.x), ord=np.inf) < 1e-8),
        "V_p": float(V_p),
        "p_th": float(p_th),
        "W_th": float(W_th),
        "P_loss": float(P_loss),
        "tau_E": float(tau_E),
        "max_residual": float(np.linalg.norm(residual(sol.x), ord=np.inf)),
        "nfev": int(sol.nfev),
        "message": str(sol.message),
    }


def main() -> None:
    values = load_vars(Path("/mnt/data/reactor.yaml"))
    R = values["R"]
    a = values["a"]
    A = values["A"]
    kappa = values.get("kappa", values["kappa_95"])
    delta = values.get("delta", values["delta_95"])
    squareness = values.get("squareness", 0.0)
    eps = a / R
    S_phi = sauter_cross_section(a, kappa, delta, squareness)
    V_p = sauter_volume(R, delta, eps, S_phi)
    n_GW = greenwald_density_limit(values["I_p"], a)
    n_avg = values["f_GW"] * n_GW
    n_la = n_avg
    p_th = thermal_pressure_uniform(n_avg, values["T_avg"])
    W_th = thermal_stored_energy(p_th, V_p)
    constants = {
        "I_p": values["I_p"],
        "B0": values["B0"],
        "n_la": n_la,
        "R": R,
        "kappa": kappa,
        "A": A,
        "afuel": values["afuel"],
        "H98_y2": values["H98_y2"],
        "P_aux_seed": values["P_aux"],
    }
    upstream = {"V_p": V_p, "p_th": p_th, "W_th": W_th}

    ordered_no_block = {**upstream, "P_loss": None, "tau_E": None, "success": False}
    ordered_block = solve_tau_block(constants, W_th)
    reconcile = solve_reconcile(constants, upstream)

    print("UPSTREAM")
    for k, v in upstream.items():
        print(f"  {k}: {v:.12g}")
    print(f"  n_GW: {n_GW:.12g}")
    print(f"  n_avg=n_la: {n_avg:.12g}")

    print("\nORDERED_NO_BLOCK")
    print(ordered_no_block)

    print("\nORDERED_WITH_BLOCK")
    for k, v in ordered_block.items():
        print(f"  {k}: {v}")

    print("\nRECONCILE")
    for k, v in reconcile.items():
        print(f"  {k}: {v}")

    print("\nCONSISTENCY")
    for label, out in [("ordered_block", ordered_block), ("reconcile", reconcile)]:
        print(f"  {label}: P_loss_MW={out['P_loss']/1e6:.9g}, tau_E={out['tau_E']:.9g}")
    print(f"  P_aux_MW={values['P_aux']/1e6:.9g}")
    print(f"  tau_scaling_at_P_aux={tau_ipb98_iter_user(P_loss=values['P_aux'], **{k:v for k,v in constants.items() if k!='P_aux_seed'}):.9g}")
    print(f"  W_th/P_aux={W_th/values['P_aux']:.9g}")


if __name__ == "__main__":
    main()
