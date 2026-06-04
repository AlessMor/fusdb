"""Tests for plasma-composition fraction and steady-state relations."""

from __future__ import annotations

from fusdb import RelationSystem, Variable
from fusdb.relations.plasma_composition.plasma_composition import (
    integrated_deuterium_fraction_from_density_profiles,
    steady_state_impurity_balance,
)


def test_deuterium_fraction_relation_is_verified_for_uniform_core_fraction():
    """Ensure the deuterium fraction relation enforces a pointwise composition."""
    # Build a profile with the same D fraction at every radial point.
    system = RelationSystem(
        [
            Variable("n_D", value=[1.0, 2.0], size=2),
            Variable("n_i", value=[2.0, 4.0], size=2),
            Variable("f_D", value=0.5),
        ],
        [integrated_deuterium_fraction_from_density_profiles],
    )

    # The direct ratio n_D / n_i should verify against the scalar core fraction.
    result = system.verify_current()
    assert result["success"] is True
    assert result["relation_status"]["Integrated D fraction from density profiles"]["status"] == "verified"


def test_deuterium_fraction_relation_detects_profile_mismatch():
    """Ensure the deuterium fraction relation rejects non-uniform core fractions."""
    # Perturb one point so the profile no longer matches a single scalar fraction.
    system = RelationSystem(
        [
            Variable("n_D", value=[1.0, 3.0], size=2),
            Variable("n_i", value=[2.0, 4.0], size=2),
            Variable("f_D", value=0.5),
        ],
        [integrated_deuterium_fraction_from_density_profiles],
    )

    # A single scalar fraction should not satisfy inconsistent pointwise ratios.
    result = system.verify_current()
    assert result["success"] is False
    assert result["relation_status"]["Integrated D fraction from density profiles"]["status"] == "violated"


def test_impurity_steady_state_balance_requires_zero_impurity_without_source():
    """Ensure the impurity steady-state residual is zero only without stored impurity."""
    # With no modeled impurity source, zero impurity should satisfy the balance.
    balanced = RelationSystem(
        [
            Variable("n_imp", value=[0.0, 0.0], size=2),
            Variable("tau_p_Imp", value=1.0),
            Variable("n_i", value=[2.0, 4.0], size=2),
        ],
        [steady_state_impurity_balance],
    ).verify_current()
    assert balanced["success"] is True
    assert balanced["relation_status"]["Steady-state Imp particle balance"]["status"] == "verified"

    # A positive impurity inventory should violate the source-free steady-state residual.
    unbalanced = RelationSystem(
        [
            Variable("n_imp", value=[1.0, 1.0], size=2),
            Variable("tau_p_Imp", value=1.0),
            Variable("n_i", value=[2.0, 4.0], size=2),
        ],
        [steady_state_impurity_balance],
    ).verify_current()
    assert unbalanced["success"] is False
    assert unbalanced["relation_status"]["Steady-state Imp particle balance"]["status"] == "violated"
