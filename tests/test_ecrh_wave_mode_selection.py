"""Contracts for explicit O/X Freethy ECCD relation selection."""

from __future__ import annotations

import numpy as np
import pytest

from fusdb.registry import (
    ELECTRON_CHARGE_C,
    ELECTRON_MASS_KG,
    EPSILON0,
    RELATIONS,
    VARIABLES,
)


def test_freethy_o_and_x_modes_are_explicit_relation_alternatives() -> None:
    values = {
        "T_e_avg": 12.0,
        "Z_eff": 1.8,
        "R": 3.0,
        "n_e_avg": 3.0e19,
        "B0": 2.1,
        "n_ecrh_harmonic": 2.0,
    }
    o_mode = RELATIONS.get("Current drive efficiency EC Freethy O-mode")
    x_mode = RELATIONS.get("Current drive efficiency EC Freethy X-mode")

    assert o_mode.input_names == x_mode.input_names
    assert "i_ecrh_wave_mode" not in VARIABLES
    assert "i_ecrh_wave_mode" not in o_mode.input_names
    assert "i_ecrh_wave_mode" not in x_mode.input_names
    assert o_mode.evaluate(values) >= 0.0
    assert x_mode.evaluate(values) >= 0.0


def test_freethy_explicit_modes_preserve_process_cutoff_formula() -> None:
    values = {
        "T_e_avg": 12.0,
        "Z_eff": 1.8,
        "R": 3.0,
        "n_e_avg": 3.0e19,
        "B0": 2.1,
        "n_ecrh_harmonic": 2.0,
    }
    fc = ELECTRON_CHARGE_C * values["B0"] / (2.0 * np.pi * ELECTRON_MASS_KG)
    fp = np.sqrt(
        values["n_e_avg"] * ELECTRON_CHARGE_C**2 / (ELECTRON_MASS_KG * EPSILON0)
    ) / (2.0 * np.pi)
    base = (
        (0.18 * 4.8 / (2.0 + values["Z_eff"]))
        * values["T_e_avg"]
        / (3.27 * values["R"] * (values["n_e_avg"] / 1.0e19))
    )
    expected_o = (
        base
        * 0.5
        * (1.0 + np.tanh(20.0 * ((values["n_ecrh_harmonic"] * fc - fp) / fp - 0.1)))
    )
    x_cutoff = 0.5 * (fc + np.sqrt(values["n_ecrh_harmonic"] * fc**2 + 4.0 * fp**2))
    expected_x = (
        base
        * 0.5
        * (
            1.0
            + np.tanh(20.0 * ((values["n_ecrh_harmonic"] * fc - x_cutoff) / fp - 0.1))
        )
    )

    assert RELATIONS.get("Current drive efficiency EC Freethy O-mode").evaluate(
        values
    ) == pytest.approx(expected_o)
    assert RELATIONS.get("Current drive efficiency EC Freethy X-mode").evaluate(
        values
    ) == pytest.approx(expected_x)
