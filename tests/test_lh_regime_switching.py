"""Confinement-regime switching at the Reactor mode boundary."""

from __future__ import annotations

import numpy as np

from fusdb.reactor import Reactor
from fusdb.registry import RELATIONS, TAGS
from fusdb.variable import Variable


H_GUARD = "H-mode sustainment (P_sep >= P_HL)"
L_GUARD = "L-mode sustainment (P_sep <= P_LH)"
L_LI_GUARD = "L-mode sustainment (P_sep <= P_LI_thresh)"
I_GUARD = "I-mode sustainment (P_sep >= P_LI_thresh)"


def _exclude_except_guards() -> tuple[str, ...]:
    keep = {H_GUARD, L_GUARD, L_LI_GUARD, I_GUARD}
    return tuple(rel.name for rel in RELATIONS if rel.name not in keep)


def _reactor(
    tag: str,
    *,
    p_sep: float = 1.0e6,
    p_lh: float = 2.0e6,
    p_li: float = 3.0e6,
    p_hl: float | None = None,
) -> Reactor:
    variables = [
        Variable("P_sep", p_sep),
        Variable("P_LH", p_lh),
        Variable("P_HL", p_hl if p_hl is not None else 0.7 * p_lh),
        Variable("P_LI_thresh", p_li),
        # Popcon axes for the scan preflight test; inert in the scalar tests.
        Variable("n_e_avg", 1.0e20),
        Variable("T_e_avg", 10.0),
    ]
    return Reactor(
        name="minimal L-H switching case",
        tags=("tokamak", tag),
        variables={var.name: var for var in variables},
        relation_exclude=_exclude_except_guards(),
    )


def test_verify_errors_when_declared_h_mode_is_below_lh_threshold() -> None:
    reactor = _reactor("h_mode")

    result = reactor.verify()

    assert not result["success"]
    assert H_GUARD in result["relation_status"]
    assert not result["relation_status"][H_GUARD]["verified"]
    assert any(H_GUARD in error for error in result["errors"])
    assert "h_mode" in reactor.tags


def test_ordered_ignores_lh_regime_guard() -> None:
    reactor = _reactor("h_mode")

    result = reactor.ordered()

    assert result["success"]
    assert result["errors"] == []
    assert result["step_status"] == []
    assert "h_mode" in reactor.tags


def test_reconcile_switches_declared_h_mode_to_l_mode_with_warning() -> None:
    reactor = _reactor("h_mode")

    result = reactor.reconcile()

    assert result["success"]
    assert "l_mode" in reactor.tags
    assert result["regime"] == "l_mode"
    assert result["regime_path"] == ["h_mode", "l_mode"]
    assert any("switched to l_mode for reconcile" in warning for warning in result["warnings"])


def test_reconcile_switches_declared_l_mode_to_i_mode_when_li_threshold_is_crossed() -> None:
    # P_sep 2.5 MW against P_LH 4.0 and P_LI 2.0: l_mode fails its L-I ceiling,
    # h_mode fails its L-H floor, i_mode is the single admissible mode.
    reactor = _reactor("l_mode", p_sep=2.5e6, p_lh=4.0e6, p_li=2.0e6)

    result = reactor.reconcile()

    assert result["success"]
    assert "i_mode" in reactor.tags
    assert result["regime"] == "i_mode"
    assert result["regime_admissible"] == ["i_mode"]
    # `regime_path` records every mode EVALUATED, in candidate order -- the
    # admissible count is the verdict, so all candidates are solved once the
    # declared mode turns out to be inadmissible.
    assert result["regime_path"] == ["l_mode", "h_mode", "i_mode"]
    assert any("switched to i_mode for reconcile" in warning for warning in result["warnings"])


# NOTE: the OTHER half of admissibility -- "the mode's own relations must hold" --
# is not exercised here.  A synthetic contradiction in this fixture gets absorbed by
# the movable inputs instead of failing a relation.  It is covered for real by
# tests/PROCESS_large_tokamak: at the tungsten point `w_1.0e-04` the h_mode solve
# fails `Energy confinement balance` and `Steady-state input-loss power balance`
# while its certifiers are moot, which is exactly the case the old walk mishandled.


def test_no_admissible_mode_is_reported_with_a_reason_per_mode() -> None:
    # P_sep below the L-H floor but above the L-I ceiling: h_mode fails its
    # certifier, l_mode fails its own, i_mode is not a candidate from h_mode.
    # Nothing is admissible, which signals an over-constrained point rather than
    # a confinement state -- and each mode must say why it was rejected.
    reactor = _reactor("h_mode", p_sep=1.0e6, p_lh=2.0e6, p_li=3.0e6)

    result = reactor.reconcile()

    assert result["regime"] == "l_mode"  # l_mode IS admissible here
    assert result["regime_admissible"] == ["l_mode"]
    assert "h_mode" in result["regime_inadmissible_reasons"]
    assert "certifiers not met" in result["regime_inadmissible_reasons"]["h_mode"]


def test_confinement_mode_axis_holds_only_transport_states() -> None:
    # Confinement mode is the plasma's transport state (does an edge barrier
    # exist), not how it is heated.  "Ohmic" is a heating method -- an ohmic
    # discharge may be L-mode or H-mode -- so it must not appear on this axis,
    # and no sustainment guard may depend on an Ohmic-L threshold power.
    assert tuple(TAGS.raw["confinement_mode"]) == ("l_mode", "h_mode", "i_mode")

    guards = [rel for rel in RELATIONS if "confinement_mode_threshold" in rel.tags]
    assert guards
    for guard in guards:
        assert "P_OL_thresh" not in guard.input_names


def test_popcon_auto_regime_assigns_l_mode_below_lh_threshold() -> None:
    # popcon now selects the confinement regime per grid point automatically
    # (no global preflight switch, no mutation of the reactor's declared tags):
    # a point below the L-H threshold (P_sep < P_LH) is solved in l_mode and
    # reported through the regime_index map, while the reactor stays h_mode.
    reactor = _reactor("h_mode")

    result = reactor.popcon(
        x={"variable": "n_e_avg", "values": [1.0e20]},
        y={"variable": "T_e_avg", "values": [10.0]},
        outputs=("P_sep", "P_LH"),
    )

    assert result["success"]
    payload = result["popcon"]
    names = payload["regime_names"]
    assert "l_mode" in names
    regime_index = payload["regime_index"]
    assert regime_index.shape == (1, 1)
    assert names[regime_index[0, 0]] == "l_mode"
    assert reactor.tags == ("tokamak", "h_mode")  # declared regime is not mutated
    assert np.isfinite(payload["fields"]["P_sep"]).all()
    assert np.isfinite(payload["fields"]["P_LH"]).all()


def test_upper_branches_are_not_mutually_reachable_until_i_mode_has_a_ceiling() -> None:
    # I<->H is physically real (I->H is a common trajectory) but is deliberately
    # NOT in the graph yet, and this test pins that as a decision rather than an
    # oversight.  MEASURED: with i_mode reachable from h_mode, the tungsten point
    # `w_1.0e-04` -- which has no consistent mode -- was classified i_mode on an
    # absurd solve (P_sep 7179.6 MW vs PROCESS 0.0 MW), because i_mode's only
    # certifier is `P_sep >= P_LI_thresh` and an inflated P_sep satisfies it.
    # Give i_mode an upper certifier (an I-H threshold) or a real H/I
    # discriminator, then open the graph and delete this test.
    from fusdb.reactor import _candidate_regimes

    assert set(_candidate_regimes("h_mode")) == {"h_mode", "l_mode"}
    assert set(_candidate_regimes("i_mode")) == {"i_mode", "l_mode"}
    assert set(_candidate_regimes("l_mode")) == {"l_mode", "h_mode", "i_mode"}
    for declared in ("h_mode", "i_mode", "l_mode"):
        assert _candidate_regimes(declared)[0] == declared


def test_order_decided_upper_branch_is_reported_as_ambiguous() -> None:
    # Declared l_mode, but P_sep clears both upper thresholds so l_mode is not
    # admissible while h_mode and i_mode both are. Nothing physical separates
    # them -- H and I differ by topology, drift direction and edge state, none
    # of which is in the decision -- so the pick must be flagged, not presented
    # as a verdict.
    reactor = _reactor("l_mode", p_sep=5.0e6, p_lh=2.0e6, p_li=1.0e6)

    result = reactor.reconcile()

    assert set(result["regime_admissible"]) == {"h_mode", "i_mode"}
    assert result.get("regime_ambiguous") is True
    assert any("NOT by a physical discriminator" in w for w in result["warnings"])


def test_hysteresis_band_makes_both_l_and_h_admissible() -> None:
    # P_sep sits BETWEEN the back-transition power and the forward threshold
    # (0.7 * 2.0 = 1.4 MW < 1.7 MW < 2.0 MW).  An existing barrier is sustained
    # there but a new one could not be created, so both modes are genuinely
    # self-consistent and history decides -- which is what hysteresis means.
    h = _reactor("h_mode", p_sep=1.7e6, p_lh=2.0e6, p_li=1.0e9)
    res_h = h.reconcile(regime_scan=True)
    assert res_h["regime"] == "h_mode"
    assert set(res_h["regime_admissible"]) == {"h_mode", "l_mode"}
    assert res_h["regime_bistable"] is True

    # Same operating point, opposite history: an L-mode machine stays L-mode.
    lo = _reactor("l_mode", p_sep=1.7e6, p_lh=2.0e6, p_li=1.0e9)
    res_l = lo.reconcile(regime_scan=True)
    assert res_l["regime"] == "l_mode"
    assert set(res_l["regime_admissible"]) == {"h_mode", "l_mode"}


def test_below_the_back_transition_power_h_mode_is_lost() -> None:
    # Below P_HL the barrier cannot be sustained however the plasma got there,
    # so a declared h_mode machine must drop to l_mode.
    reactor = _reactor("h_mode", p_sep=1.0e6, p_lh=2.0e6, p_li=1.0e9)

    result = reactor.reconcile()

    assert result["regime"] == "l_mode"
    assert result["regime_admissible"] == ["l_mode"]
