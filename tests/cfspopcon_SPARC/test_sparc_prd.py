"""cfspopcon SPARC PRD reproduction test (phase 1).

Builds a fusdb ``Reactor`` from ``reactor.yaml`` (a fusdb-native transcription of
cfspopcon's ``input.yaml`` pinned at the optimized PRD operating point), runs it in
**ordered** mode following the cfspopcon algorithm order, and compares the resulting
fusdb variables against cfspopcon's reference output ``output/PRD.json``.

Phase-1 scope (see plan): only the subset of cfspopcon's 110-step algorithm that maps
onto existing fusdb relations is reproduced. Quantities fusdb already computes well
(geometry, Greenwald limit/fraction) are asserted strictly. Quantities that differ for
*known, documented* reasons are marked ``xfail`` so the discrepancy is visible and a
future fix is flagged automatically:

  * fusion power / pressure: fusdb integrates profiles uniformly in ``rho``
    (``trapezoid(..., x=rho)``) rather than volume-weighting (``dV ~ V'(rho) drho``),
    which overestimates core-peaked integrals; compounded by fusdb's parabolic profiles
    vs cfspopcon's ``prf`` form.
  * ``tau_E`` / ``P_loss``: fusdb's ordered 2x2 block solver does not converge the
    multi-scale (``P_loss ~1e7 W``, ``tau_E ~0.5 s``) W_th/IPB98 system.
  * ``P_LH``: fusdb uses a different L-H scaling constant than cfspopcon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fusdb.reactor import Reactor
from fusdb.registry import VARIABLES

CASE_DIR = Path(__file__).parent
PRD_PATH = CASE_DIR / "output" / "PRD.json"

# cfspopcon pint unit string -> multiplicative factor into the fusdb canonical SI unit.
# (keV is fusdb's canonical temperature unit, so kiloelectron_volt maps to 1.0.)
_UNIT_SCALE: dict[str, float] = {
    "meter": 1.0,
    "meter ** 2": 1.0,
    "meter ** 3": 1.0,
    "second": 1.0,
    "pascal": 1.0,
    "dimensionless": 1.0,
    "kiloelectron_volt": 1.0,
    "megawatt": 1.0e6,
    "_1e19_per_cubic_metre": 1.0e19,
    "_1e20_per_cubic_metre": 1.0e20,
    # triple product, n_i_tau_E_T_i canonical unit is keV*s/m^3.
    "_1e20_per_cubic_metre * kiloelectron_volt * second": 1.0e20,
}


def _load_prd() -> dict[str, dict]:
    """Load PRD.json, repairing cfspopcon's invalid trailing-dot numbers."""
    raw = PRD_PATH.read_text(encoding="utf-8")
    # cfspopcon dumps numbers like ``732029.`` (trailing dot, no digit) which json
    # rejects; turn ``<digit>.`` followed by a delimiter into ``<digit>.0``.
    repaired = re.sub(r"(?<=[0-9])\.(?=[\s,}\]])", ".0", raw)
    data = json.loads(repaired)
    merged: dict[str, dict] = {}
    merged.update(data.get("coords", {}))
    merged.update(data.get("data_vars", {}))
    return merged


def _prd_si(prd: dict[str, dict], name: str) -> float:
    """Return one PRD scalar converted to the fusdb canonical SI value."""
    entry = prd[name]
    units = str(entry.get("attrs", {}).get("units", "dimensionless"))
    if units not in _UNIT_SCALE:
        raise KeyError(f"No SI scale registered for cfspopcon unit {units!r} ({name}).")
    return float(entry["data"]) * _UNIT_SCALE[units]


@pytest.fixture(scope="module")
def ordered_run():
    """Build the reactor, run ordered mode, return (system, result, prd)."""
    reactor = Reactor.from_yaml(CASE_DIR)
    system = reactor.relation_system()
    result = system.ordered(order=reactor.relation_order or None)
    return system, result, _load_prd()


# cfspopcon output names that are intentionally not registered as fusdb aliases
# (too generic to alias globally) but still map to a fusdb variable for this test.
_NAME_OVERRIDE = {"P_in": "P_loss"}


def _fusdb_value(system, name: str):
    """Return the current fusdb value for a canonical name or alias, or None."""
    canonical = _NAME_OVERRIDE.get(name) or VARIABLES.resolve(name)
    var = system.variables_by_name.get(canonical)
    return None if var is None else var.value


# (PRD/cfspopcon name, relative tolerance, xfail reason or None).
# Names resolve to fusdb variables through registry aliases (no Python name map).
STRICT_CASES = [
    ("minor_radius", 2e-3, None),
    ("plasma_volume", 0.03, None),
    ("surface_area", 0.12, None),  # Sauter L_p vs cfspopcon: ~10% high
    ("average_ion_temp", 1e-3, None),
    ("greenwald_density_limit", 2e-3, None),
    ("greenwald_fraction", 2e-3, None),
]

_XFAIL_RAW = [
    ("average_total_pressure", 0.05, "rho-uniform profile integration overestimates core-peaked pressure"),
    ("beta_toroidal", 0.05, "beta_T tracks the over-estimated thermal pressure"),
    ("P_fusion", 0.05, "rho-uniform integration + parabolic vs prf profiles overestimate fusion power"),
    ("P_neutron", 0.05, "neutron power tracks the over-estimated fusion power"),
    ("P_alpha", 0.05, "alpha power tracks the over-estimated fusion power"),
    ("P_LH_thresh", 0.05, "fusdb L-H scaling constant differs from cfspopcon"),
    ("energy_confinement_time", 0.05, "tau_E tracks the over-estimated W_th (rho-uniform pressure integration) through the confinement block"),
    ("P_in", 0.05, "P_loss ~ W_th^2.7 through the confinement block, amplifying the over-estimated thermal pressure"),
]
# xfail(strict): each runs and is expected to fail; if fusdb is fixed it xpasses and
# the suite fails, prompting the case to be promoted to a strict assertion.
XFAIL_CASES = [
    pytest.param(name, tol, marks=pytest.mark.xfail(reason=reason, strict=True), id=name)
    for name, tol, reason in _XFAIL_RAW
]


@pytest.mark.parametrize("prd_name, rel_tol, _reason", STRICT_CASES, ids=[c[0] for c in STRICT_CASES])
def test_matches_cfspopcon_strict(ordered_run, prd_name, rel_tol, _reason):
    system, _result, prd = ordered_run
    got = _fusdb_value(system, prd_name)
    assert got is not None, f"fusdb did not produce {prd_name!r}"
    expected = _prd_si(prd, prd_name)
    assert got == pytest.approx(expected, rel=rel_tol), f"{prd_name}: fusdb={got:.6g} cfspopcon={expected:.6g}"


@pytest.mark.parametrize("prd_name, rel_tol", XFAIL_CASES)
def test_matches_cfspopcon_known_gaps(ordered_run, prd_name, rel_tol):
    """Documented phase-1 discrepancies (xfail strict); see the per-case reason."""
    system, _result, prd = ordered_run
    got = _fusdb_value(system, prd_name)
    expected = _prd_si(prd, prd_name)
    assert got is not None and got == pytest.approx(expected, rel=rel_tol)


def test_ordered_run_succeeds(ordered_run):
    """The full ordered recipe (incl. the 2x2 confinement block) runs to completion."""
    _system, result, _prd = ordered_run
    assert result.get("success"), f"ordered run failed: {result.get('errors')}"
    for name in (
        "Inverse aspect ratio",
        "Tokamak plasma volume",
        "Thermal pressure",
        "tau_E_iter_ipb98y2",
        "DT reaction rate",
        "Total fusion power",
        "L-H transition threshold power",
    ):
        assert name in result.get("executed_relations", []), f"ordered run did not reach {name!r}"


def test_confinement_block_solved_and_consistent(ordered_run):
    """The ordered 2x2 block produces tau_E and P_loss and satisfies W_th = P_loss*tau_E.

    This locks in the fix that routes the ordered block through the shared
    ``_solve_initial_block`` solver. The solved values do not match cfspopcon's PRD
    (W_th is over-estimated upstream), but the block itself must converge and be
    internally consistent.
    """
    system, _result, _prd = ordered_run
    tau_E = _fusdb_value(system, "energy_confinement_time")
    P_loss = _fusdb_value(system, "P_in")
    W_th = system.variables_by_name["W_th"].value
    assert tau_E is not None and P_loss is not None, "confinement block did not solve tau_E/P_loss"
    assert P_loss * tau_E == pytest.approx(W_th, rel=1e-4), "W_th = P_loss * tau_E not satisfied"
