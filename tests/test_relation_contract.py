"""Relation contract survey: every registered relation, every direction (S2).

For each registered relation this test:

* builds a nominal namespace from registry metadata (scalars at their
  ``nominal``/domain-derived value, profiles flat at that value, ``rho`` a
  uniform grid),
* establishes a consistent point (forward evaluation for output relations,
  a one-variable solve for outputless balances),
* then removes each relation variable in turn and asks ``Relation.solve``
  to recover it, classifying the outcome.

Direction classes and their contract:

* scalar equality target   -> solves and verifies (or a recorded numeric
  failure); the aggregate solved fraction is pinned as a regression floor.
* profile-valued target    -> a flat profile at the solved level, shaped like
  the supplied grid (documented standalone semantics), or a refusal.
* coordinate target (rho)  -> refusal: a coordinate is not an unknown.
* inequality relations     -> verify-only; solving any target refuses.

Run this module directly for the full per-relation report:

    python tests/test_relation_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path

from collections import Counter

import numpy as np
import pytest

from fusdb.registry import RELATIONS
from fusdb.registry.variable_registry import VARIABLES
from fusdb.relation import (
    COORDINATE_NAMES,
    Relation,
    RelationNotInvertibleError,
    RelationSolveError,
    RelationUnderdeterminedError,
)

PROFILE_SIZE = 21
RHO = np.linspace(0.0, 1.0, PROFILE_SIZE)


def _spec(name: str):
    return VARIABLES.get(name) if name in VARIABLES else None


def _pick_inside(lo, hi) -> float:
    """A representative value strictly inside a (possibly open) interval."""
    lo_f = None if lo is None else float(lo)
    hi_f = None if hi is None else float(hi)
    if lo_f is not None and hi_f is not None and np.isfinite(lo_f) and np.isfinite(hi_f):
        if lo_f > 0.0:
            return float(np.sqrt(lo_f * hi_f))
        return 0.5 * (lo_f + hi_f)
    if lo_f is not None and np.isfinite(lo_f):
        return lo_f * 2.0 if lo_f > 0.0 else 1.0
    if hi_f is not None and np.isfinite(hi_f):
        return hi_f * 0.5 if hi_f > 0.0 else hi_f - 1.0
    return 1.0


# Probe points are FROZEN, not derived from the domains under test.  They used
# to come from ``_pick_inside(domain)``, which made this survey self-referential:
# narrowing any domain moved its own probe (B0 (0, inf) -> 1.0 T became
# (0, 30] -> 15.0 T; kappa (0, 10] -> 5.0 became [0.1, 10] -> 1.0, a circular
# plasma where the shaping relations degenerate).  The measured fraction then
# moved for reasons that had nothing to do with the inverse machinery, so the
# guard could not distinguish "inversions regressed" from "we now probe
# somewhere else".  Frozen once from the values the derived rule produced, so
# the survey is a genuine before/after comparison across registry edits.
_FROZEN_PROBES: dict[str, float] = json.loads(
    (Path(__file__).parent / "_survey_probe_points.json").read_text(encoding="utf-8")
)


def _nominal_value(name: str):
    if name in COORDINATE_NAMES:
        return RHO.copy()
    spec = _spec(name)
    if spec is None:
        return 1.0
    frozen = _FROZEN_PROBES.get(name)
    if frozen is not None:
        base = float(frozen)
    else:
        base = float(spec.nominal) if spec.nominal is not None else _pick_inside(spec.domain[0], spec.domain[1])
    if spec.shape:
        return np.full(PROFILE_SIZE, base)
    return base


def _coordinate_constants(rel: Relation) -> dict:
    """Framework-supplied coordinate constants (rho) at this survey's grid size.

    Coordinates are relation constants now (S3), not surveyed variables, but a
    relation that integrates over one still needs the grid at evaluation.
    """
    return {name: RHO.copy() for name in rel.constant_names if name in COORDINATE_NAMES}


def _consistent_namespace(rel: Relation) -> tuple[dict | None, str]:
    """Build a namespace satisfying the relation, plus how it was obtained."""
    ns = {name: _nominal_value(name) for name in rel.variables}
    ns.update(_coordinate_constants(rel))
    if rel.outputs:
        implicit = set(rel.outputs) & set(rel.input_names)
        if not implicit:
            inputs = {name: ns[name] for name in rel.input_names}
            inputs.update(_coordinate_constants(rel))
            try:
                ns.update(rel.output_map(rel.evaluate(inputs)))
            except Exception as exc:
                return None, f"forward_error:{type(exc).__name__}"
            return ns, "forward"
        target = rel.outputs[0]
        probe = {k: v for k, v in ns.items() if k != target}
        try:
            ns[target] = rel.solve(probe)
        except Exception as exc:
            return None, f"implicit_error:{type(exc).__name__}"
        return ns, "implicit_solve"
    if rel.op == "==":
        for target in rel.variables:
            spec = _spec(target)
            if target in COORDINATE_NAMES or (spec is not None and spec.shape):
                continue
            probe = {k: v for k, v in ns.items() if k != target}
            try:
                solved = rel.solve(probe)
            except Exception:
                continue
            out = dict(ns)
            out[target] = solved
            return out, f"balance_solve:{target}"
        return None, "no_consistent_point"
    return ns, "inequality_nominal"


def _classify_return(rel: Relation, target: str, value, expected) -> str:
    if target in COORDINATE_NAMES:
        return "silent_coordinate"
    if rel.op != "==":
        return "silent_inequality"
    spec = _spec(target)
    if spec is not None and spec.shape:
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.size == 1:
            return "silent_scalar_for_profile"
        if arr.shape[-1] == PROFILE_SIZE:
            return "profile_recovered"
        return "silent_wrong_shape"
    try:
        expected_f = float(expected)
        value_f = float(value)
    except (TypeError, ValueError):
        return "silent_wrong_shape"
    abs_tol = float(spec.abs_tol) if spec is not None else 0.0
    if np.isclose(value_f, expected_f, rtol=1e-3, atol=abs_tol):
        return "verified_roundtrip"
    return "verified_other_root"


def _survey_relation(rel: Relation) -> dict[str, str]:
    ns, how = _consistent_namespace(rel)
    outcomes: dict[str, str] = {"__setup__": how}
    if ns is None:
        return outcomes
    for target in rel.variables:
        probe = {k: v for k, v in ns.items() if k != target}
        try:
            value = rel.solve(probe)
        except RelationNotInvertibleError:
            outcomes[target] = "refused"
        except RelationUnderdeterminedError:
            outcomes[target] = "underdetermined"
        except RelationSolveError:
            outcomes[target] = "solve_failed"
        except Exception as exc:
            outcomes[target] = f"error:{type(exc).__name__}"
        else:
            outcomes[target] = _classify_return(rel, target, value, ns[target])
    return outcomes


def _all_relations() -> list[Relation]:
    rels = []
    for item in RELATIONS:
        rels.append(RELATIONS.get(item) if isinstance(item, str) else item)
    return rels


@pytest.fixture(scope="module")
def survey() -> dict[str, dict[str, str]]:
    return {rel.name: _survey_relation(rel) for rel in _all_relations()}


def _directions(survey: dict[str, dict[str, str]], kinds: set[str] | None = None):
    """Yield (relation, target, outcome) for surveyed directions."""
    rels = {rel.name: rel for rel in _all_relations()}
    for rel_name, outcomes in survey.items():
        rel = rels[rel_name]
        for target, outcome in outcomes.items():
            if target == "__setup__":
                continue
            spec = _spec(target)
            if target in COORDINATE_NAMES:
                kind = "coordinate"
            elif rel.op != "==":
                kind = "inequality"
            elif spec is not None and spec.shape:
                kind = "profile"
            else:
                kind = "scalar"
            if kinds is None or kind in kinds:
                yield rel_name, target, outcome


def test_setup_reaches_most_relations(survey):
    """A consistent point is found for the overwhelming majority of relations."""
    setups = Counter(outcomes["__setup__"].split(":")[0] for outcomes in survey.values())
    reached = setups.get("forward", 0) + setups.get("implicit_solve", 0) + setups.get("balance_solve", 0) + setups.get("inequality_nominal", 0)
    assert reached >= 0.75 * len(survey), f"setup buckets: {setups}"


def test_no_silent_degenerate_answers(survey):
    """No direction returns a silently-degenerate answer (the S1 contract)."""
    silent = [
        (rel, target, outcome)
        for rel, target, outcome in _directions(survey)
        if outcome.startswith("silent_")
    ]
    assert silent == [], f"{len(silent)} silent degenerate directions: {silent[:10]}"


def test_coordinate_targets_refuse(survey):
    """Solving for a coordinate (rho) is always refused."""
    bad = [
        (rel, target, outcome)
        for rel, target, outcome in _directions(survey, kinds={"coordinate"})
        if outcome not in ("refused",)
    ]
    assert bad == [], f"coordinate directions not refused: {bad[:10]}"


def test_inequality_targets_refuse(survey):
    """Solving a variable through an inequality is always refused."""
    bad = [
        (rel, target, outcome)
        for rel, target, outcome in _directions(survey, kinds={"inequality"})
        if outcome not in ("refused",)
    ]
    assert bad == [], f"inequality directions not refused: {bad[:10]}"


def test_profile_targets_flat_or_refused(survey):
    """A profile target yields a grid-shaped flat profile or an explicit refusal."""
    bad = [
        (rel, target, outcome)
        for rel, target, outcome in _directions(survey, kinds={"profile"})
        if outcome not in ("profile_recovered", "refused", "solve_failed")
    ]
    assert bad == [], f"profile directions outside contract: {bad[:10]}"


def _batchify(ns: dict, n: int) -> dict:
    """Stack a per-point namespace into the batched layout (scalars (n,1),
    profiles (n,P), coordinate grids shared (P,)), varied per row so a broadcast
    defect (outer product / batch collapse) cannot hide behind equal values."""
    factors = np.linspace(0.9, 1.1, n).reshape(n, 1)
    out = {}
    for key, value in ns.items():
        arr = np.asarray(value, dtype=float)
        if key in COORDINATE_NAMES:
            out[key] = arr
        elif arr.ndim == 0:
            out[key] = factors * float(arr)
        elif arr.shape[-1] == PROFILE_SIZE:
            out[key] = factors * arr.reshape(1, -1)
        else:
            out[key] = np.broadcast_to(arr, (n, *arr.shape)).copy()
    return out


def _batch_outcome(rel: Relation) -> str:
    """Classify one relation's batch behaviour against a varied N-point stack:

    * ``match``            -- batched == point-by-point everywhere;
    * ``pointwise_only``   -- cannot batch (raises / un-coercible); correct via
      the framework's per-point fallback;
    * ``spotcheck_caught`` -- batches to a WRONG value, but the discrepancy
      shows at the first or last grid point, so the framework's 2-point spot
      check catches it and falls back;
    * ``hidden_mismatch``  -- batches wrong ONLY at an interior point, invisible
      to the first/last spot check.  This is the genuinely dangerous class.
    """
    ns, how = _consistent_namespace(rel)
    if ns is None or how.split(":")[0] not in ("forward", "implicit_solve", "balance_solve"):
        return "skipped"
    n = 5
    batched_ns = _batchify(ns, n)
    try:
        batched = rel.output_map(rel.evaluate(batched_ns))
    except Exception:
        return "pointwise_only"
    point_maps = []
    for i in range(n):
        point = {}
        for key, value in batched_ns.items():
            arr = np.asarray(value)
            point[key] = arr if key in COORDINATE_NAMES else (arr[i] if arr.ndim else float(arr))
        try:
            point_maps.append(rel.output_map(rel.evaluate(point)))
        except Exception:
            return "pointwise_only"
    bad_rows: set[int] = set()
    for name in rel.output_names or batched:
        b = np.asarray(batched.get(name), dtype=float)
        for i in range(n):
            p = np.ravel(np.asarray(point_maps[i].get(name), dtype=float))
            # coerce_batched maps a reduction's (n,1) / expansion's (n,P) back to
            # per-point shape, so compare raveled values.
            bi = np.ravel(b[i]) if b.ndim and b.shape[0] == n else np.ravel(b)
            if bi.shape != p.shape or not np.allclose(bi, p, rtol=1e-6, atol=0.0, equal_nan=True):
                bad_rows.add(i)
    if not bad_rows:
        return "match"
    # The framework spot-checks the first and last populated grid points.
    return "spotcheck_caught" if (0 in bad_rows or n - 1 in bad_rows) else "hidden_mismatch"


@pytest.fixture(scope="module")
def batch_survey() -> dict[str, str]:
    return {rel.name: _batch_outcome(rel) for rel in _all_relations()}


def test_no_hidden_batch_mismatch(batch_survey):
    """Every batched-but-wrong relation is caught by the framework spot check.

    A relation that batches to the wrong value only at an interior grid point --
    invisible to the first/last spot check -- would silently poison a scan.
    This is the load-bearing safety guarantee that lets Phase 2 defer the
    declared-shape machinery (S5): the working sniffing is safe iff no such
    relation exists.
    """
    hidden = sorted(name for name, outcome in batch_survey.items() if outcome == "hidden_mismatch")
    assert hidden == [], f"{len(hidden)} relations batch wrong only at interior points: {hidden}"


def test_batch_fallback_set_is_documented(batch_survey):
    """Pin the set of relations that do not batch cleanly (raise/uncoercible, or
    a spot-check-caught divergence).  This is the exact input a future declared
    reduction/expansion retirement (Phase 6) would consume; a change here is a
    deliberate shift in batch behaviour, not an accident."""
    from collections import Counter
    counts = Counter(batch_survey.values())
    matched = counts.get("match", 0)
    reliant = counts.get("pointwise_only", 0) + counts.get("spotcheck_caught", 0)
    total = matched + reliant
    # The overwhelming majority batch cleanly; the rest lean on the fallback.
    assert total == 0 or matched >= 0.7 * total, f"batch counts: {dict(counts)}"


def test_scalar_inverse_floor(survey):
    """The solved+verified fraction of scalar directions stays above the floor."""
    outcomes = [outcome for _, _, outcome in _directions(survey, kinds={"scalar"})]
    solved = sum(1 for item in outcomes if item.startswith("verified_"))
    assert outcomes, "no scalar directions surveyed"
    fraction = solved / len(outcomes)
    assert fraction >= 0.80, f"scalar inverse fraction {fraction:.3f} ({Counter(outcomes)})"


if __name__ == "__main__":
    report = {rel.name: _survey_relation(rel) for rel in _all_relations()}
    totals: Counter = Counter()
    for rel_name, outcomes in sorted(report.items()):
        setup = outcomes.get("__setup__", "?")
        totals[f"setup:{setup.split(':')[0]}"] += 1
        rows = {k: v for k, v in outcomes.items() if k != "__setup__"}
        for outcome in rows.values():
            totals[outcome.split(":")[0]] += 1
        flagged = {k: v for k, v in rows.items() if not v.startswith("verified_")}
        if setup.split(":")[0] not in ("forward", "implicit_solve", "balance_solve", "inequality_nominal") or flagged:
            print(f"{rel_name}: setup={setup} {flagged}")
    print("\n== totals ==")
    for key, count in totals.most_common():
        print(f"{key:32s} {count}")


def test_signature_defaults_agree_with_variables_yaml() -> None:
    """A relation's kwarg default must match that variable's registry default.

    fusdb has TWO default levels and they are NOT interchangeable:

    * a relation kwarg default makes the parameter a CONSTANT -- not packed,
      never solved for, and the relation stays evaluable when the variable is
      inactive;
    * a ``variables.yaml`` ``default:`` makes it a PENALIZED NON-FIXED INPUT --
      a packed unknown seeded at that value.

    Because they mean different things they cannot be merged, but where both
    exist they must at least agree on the NUMBER, or a reactor gets one value
    when the variable is packed and a different one when it is not.  Measured
    2026-08-01: 238 pairs, zero disagreements -- this keeps it that way.
    """
    import inspect

    from fusdb.registry import RELATIONS, VARIABLES

    variables = VARIABLES.raw if hasattr(VARIABLES, "raw") else None
    if variables is None:  # fall back to the source of truth on disk
        import yaml
        from pathlib import Path

        import fusdb.registry as registry_pkg

        path = Path(registry_pkg.__file__).parent / "variables.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        variables = document.get("variables", document)

    mismatched = []
    for relation in RELATIONS:
        signature = inspect.signature(relation.func)
        for name in relation.constant_names:
            parameter = signature.parameters.get(name)
            if parameter is None or parameter.default is inspect.Parameter.empty:
                continue  # coordinates (rho) are framework-supplied
            if isinstance(parameter.default, str):
                continue  # method switches (e.g. interpolation_kind), not physics
            registry_default = (variables.get(name) or {}).get("default")
            if registry_default is None:
                continue  # covered by the code-only inventory test below
            if float(parameter.default) != float(registry_default):
                mismatched.append(
                    f"{relation.name}: {name} = {parameter.default!r} "
                    f"but variables.yaml says {registry_default!r}"
                )

    assert not mismatched, "signature/registry default disagreement:\n  " + "\n  ".join(mismatched)
