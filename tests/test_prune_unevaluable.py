"""Compilation evaluates, holds, or prunes composition by what is determinable.

A variable is applied from its registry ``default`` (``variables.yaml``) as either
a held constant (when nothing can move it) or a balance-driven free core (when a
constraint determines it).  The He3/He4/impurity ash fractions default to 0 and
are gated on ``tau_p`` (``default_requires``):

* Without ``tau_p`` the steady-state particle balances cannot activate (their
  ``tau_p`` is a pure input the structural partition leaves underdetermined), so
  the ash fractions are held at 0 as derived constants -- not packed as solver
  unknowns and not invented as free knobs.
* With ``tau_p`` the balances activate and the fuel/ash fractions become free
  cores the balance moves, deriving the trace He ash.

Genuinely unevaluable variables (no supply, no producer, no determining block)
are still pruned together with the relations that need them.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from fusdb.reactor import Reactor

REACTORS = sorted(glob.glob("reactors/*/reactor.yaml"))
REACTOR_IDS = [os.path.basename(os.path.dirname(p)) for p in REACTORS]


@pytest.mark.parametrize("path", REACTORS, ids=REACTOR_IDS)
def test_no_required_uninitialized_free_variables(path: str) -> None:
    """Every shipped reactor compiles with no unevaluable required free vars.

    This is the guarantee that reconcile initialization does not fail: any
    variable that could not be evaluated has been pruned, held at a default, or
    determined, never left as a required-but-uninitialized free variable.
    """
    system = Reactor.from_yaml(path).relation_system()
    system._pack_free_variables()
    assert system._required_uninitialized_free_variables() == []
    assert "P_fus" in system.active_variable_names


def test_pure_dt_holds_he_at_zero_without_tau_p() -> None:
    """With no ``tau_p`` the He ash fractions are held at 0, not solved."""
    system = Reactor.from_yaml("reactors/STEP_2024/reactor.yaml").relation_system()
    constants = system.constant_default_values
    # Ash fractions are held constants at their default 0, not free unknowns.
    for name in ("f_He3", "f_He4", "f_Imp"):
        assert name in constants
        assert float(constants[name]) == 0.0
        assert name not in system.block_core_names
        assert name in system.derived_variable_names
    # The fuel fractions are likewise held at the equimolar default.
    for name in ("f_D", "f_T"):
        assert float(constants[name]) == 0.5
    # No particle balance activates without a confinement time to pin it.
    assert not any(
        "particle balance" in name.lower()
        for name in system.compiler_report["active_relations"]
    )


def test_he_ash_derived_with_tau_p() -> None:
    """With ``tau_p`` the balances activate and the trace He ash is derived."""
    system = Reactor.from_yaml("reactors/ARC_V0/reactor.yaml").relation_system()
    active = set(system.compiler_report["active_relations"])
    # All four steady-state balances are active and the ash fractions are free
    # cores the balance moves (not held constants).
    assert {
        "Steady-state D particle balance",
        "Steady-state T particle balance",
        "Steady-state He3 particle balance",
        "Steady-state He4 particle balance",
    } <= active
    assert "f_He4" in system.block_core_names
    assert "f_He4" not in system.constant_default_values

    result = system.reconcile()
    variables = result["variables"]
    f_He4 = float(np.asarray(variables["f_He4"].value, dtype=float).mean())
    f_D = float(np.asarray(variables["f_D"].value, dtype=float).mean())
    f_T = float(np.asarray(variables["f_T"].value, dtype=float).mean())
    # Trace He ash is produced, and the fuel split stays near-symmetric ~0.5.
    assert 1e-4 < f_He4 < 0.1
    assert abs(f_D - f_T) < 0.05
    assert 0.45 < f_D < 0.5
