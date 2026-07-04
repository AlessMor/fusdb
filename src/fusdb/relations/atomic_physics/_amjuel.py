"""AMJUEL H.2 and H.4 rate-coefficient data loading and evaluation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_KEV_TO_EV = 1.0e3
_CM3_S_TO_M3_S = 1.0e-6
_CM3_PER_M3 = 1.0e6
_DENSITY_SCALE_CM3 = 1.0e8


def _parse_fortran_float(value: str) -> float:
    text = str(value).strip().replace("D", "E").replace("d", "E")
    text = text.replace("E ", "E+").replace("e ", "E+")
    return float(text)


@lru_cache(maxsize=None)
def load_amjuel_h2_fit(path: str | Path) -> dict[str, Any]:
    """Load one AMJUEL H.2 YAML coefficient file."""
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    coefficients = data.get("coefficients", ())
    if len(coefficients) != 9:
        raise ValueError(f"Expected 9 AMJUEL H.2 coefficients in {resolved}.")
    parsed = np.array([_parse_fortran_float(value) for value in coefficients], dtype=float)
    return {**data, "path": resolved, "parsed_coefficients": parsed}


def evaluate_amjuel_h2_rate(path: str | Path, T_edge: Any) -> Any:
    """Evaluate an AMJUEL H.2 fit and return the rate coefficient in m^3/s."""
    fit = load_amjuel_h2_fit(path)
    T_eV = np.asarray(T_edge, dtype=float) * _KEV_TO_EV
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        log_T = np.log(T_eV)
        exponent = np.zeros(np.shape(log_T), dtype=float)
        for index, coefficient in enumerate(fit["parsed_coefficients"]):
            exponent = exponent + coefficient * (log_T**index)
        rate = np.exp(exponent) * _CM3_S_TO_M3_S
    if rate.shape == ():
        return float(rate)
    return rate


@lru_cache(maxsize=None)
def load_amjuel_h4_fit(path: str | Path) -> dict[str, Any]:
    """Load one AMJUEL H.4 YAML coefficient file."""
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    blocks = data.get("coefficient_blocks", ())
    coefficients = np.zeros((9, 9), dtype=float)
    for block in blocks:
        density_indices = [int(item) for item in block["density_indices"]]
        for row in block["rows"]:
            temperature_index = int(row["temperature_index"])
            values = row["coefficients"]
            if len(values) != len(density_indices):
                raise ValueError(f"Bad AMJUEL row width in {resolved}.")
            for density_index, value in zip(density_indices, values):
                coefficients[temperature_index, int(density_index)] = _parse_fortran_float(value)
    return {**data, "path": resolved, "coefficients": coefficients}


def evaluate_amjuel_h4_rate(path: str | Path, n_e_edge: Any, T_edge: Any) -> Any:
    """Evaluate an AMJUEL H.4 fit and return the rate coefficient in m^3/s."""
    fit = load_amjuel_h4_fit(path)
    density_limits = fit["density_limits"]
    n_cm3 = np.asarray(n_e_edge, dtype=float) / _CM3_PER_M3
    n_min = _parse_fortran_float(density_limits["min_cm3"])
    n_max = _parse_fortran_float(density_limits["max_cm3"])
    n_tilde = np.clip(n_cm3, n_min, n_max) / _DENSITY_SCALE_CM3
    T_eV = np.asarray(T_edge, dtype=float) * _KEV_TO_EV
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        log_n = np.log(n_tilde)
        log_T = np.log(T_eV)
        exponent = np.zeros(np.broadcast_shapes(np.shape(log_n), np.shape(log_T)), dtype=float)
        log_n_b = np.broadcast_to(log_n, exponent.shape)
        log_T_b = np.broadcast_to(log_T, exponent.shape)
        for temperature_index in range(9):
            T_power = log_T_b**temperature_index
            for density_index in range(9):
                exponent = exponent + fit["coefficients"][temperature_index, density_index] * T_power * (log_n_b**density_index)
        rate = np.exp(exponent) * _CM3_S_TO_M3_S
    if rate.shape == ():
        return float(rate)
    return rate
