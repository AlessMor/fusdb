"""Profile <-> volume-average consistency is enforced by a real relation.

A supplied profile and its scalar average are linked by an outputless
``<x>_avg == volume_average(<x>)`` residual rather than a build-time override:
if a fixed profile and a separately supplied average disagree, the mode surfaces
it (verify) or reconciles it (reconcile), instead of silently overwriting the
average and warning at construction time.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusdb.registry import RELATIONS
from fusdb.relationsystem import RelationSystem
from fusdb.variable import Variable

_CONSISTENCY = "Electron temperature volume-average consistency"


def _system(t_e_avg: float, profile: np.ndarray, *, avg_fixed: bool = False) -> RelationSystem:
    """A minimal system: a fixed T_e profile plus a supplied scalar T_e_avg."""
    rho = np.linspace(0.0, 1.0, profile.size)
    rel = RELATIONS.get(_CONSISTENCY)
    variables = [
        Variable("rho", value=rho, fixed=True),
        Variable("T_e", value=profile, fixed=True),
        Variable("T_e_avg", value=t_e_avg, fixed=avg_fixed),
    ]
    return RelationSystem(variables, [rel], name="profile_average_test")


def test_fixed_profile_conflicting_average_is_flagged_on_verify():
    profile = np.full(21, 15.0)  # volume-average 15
    system = _system(14.0, profile)  # supplied average disagrees
    result = system.run("verify")
    assert not result["verified"]
    assert not result["relation_status"][_CONSISTENCY]["verified"]


def test_fixed_profile_consistent_average_passes_verify():
    profile = np.full(21, 15.0)
    system = _system(15.0, profile)
    result = system.run("verify")
    assert result["relation_status"][_CONSISTENCY]["verified"]


def test_reconcile_moves_supplied_average_to_the_profile_value():
    profile = np.full(21, 15.0)
    system = _system(14.0, profile)  # not fixed: reconcile is free to move it
    result = system.run("reconcile")
    assert result["verified"]
    assert system.values["T_e_avg"] == pytest.approx(15.0, abs=1e-3)


def test_shape_locked_profile_residual_is_trivially_satisfied():
    """A level-free profile reconstructed as avg*shape never fights the residual."""
    rho = np.linspace(0.0, 1.0, 51)
    shape = (1.0 - rho**2) ** 1.5 + 0.1
    profile = 8.0 * shape  # unfixed supplied profile: shape locked, level free
    variables = [
        Variable("rho", value=rho, fixed=True),
        Variable("T_e", value=profile, fixed=False),
    ]
    system = RelationSystem(variables, [RELATIONS.get(_CONSISTENCY)], name="level_free_test")
    result = system.run("verify")
    assert result["relation_status"][_CONSISTENCY]["verified"]
