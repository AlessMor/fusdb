"""Confinement-regime switching at the Reactor mode boundary."""

from __future__ import annotations

import numpy as np

from fusdb.reactor import Reactor
from fusdb.registry import RELATIONS, TAGS
from fusdb.variable import Variable


H_GUARD = "H-mode sustainment (P_sep >= P_LH)"
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
) -> Reactor:
    variables = [
        Variable("P_sep", p_sep),
        Variable("P_LH", p_lh),
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
    reactor = _reactor("l_mode", p_sep=2.5e6, p_lh=4.0e6, p_li=2.0e6)

    result = reactor.reconcile()

    assert result["success"]
    assert "i_mode" in reactor.tags
    assert result["regime"] == "i_mode"
    assert result["regime_path"] == ["l_mode", "i_mode"]
    assert any("switched to i_mode for reconcile" in warning for warning in result["warnings"])


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
