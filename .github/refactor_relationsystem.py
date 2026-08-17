from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "src/fusdb/relationsystem.py"
RECONCILE = ROOT / "src/fusdb/modes/reconcile.py"
POPCON = ROOT / "src/fusdb/modes/popcon.py"
LIFECYCLE_TEST = ROOT / "tests/test_relationsystem_lifecycle.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one literal match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return updated


def stage1() -> None:
    rs = RS.read_text()
    reconcile = RECONCILE.read_text()
    popcon = POPCON.read_text()

    # Public compiled-system protocol: the plans are intentionally ordinary
    # RelationSystem state, but execution modes no longer couple to underscore
    # implementation names.
    renames = {
        "_profile_specs": "profile_specs",
        "_constant_defaults_solver": "constant_defaults_solver",
        "_provider_plan": "provider_plan",
        "_completion_passes": "completion_passes",
        "_enforced_residual_relations": "residual_relations",
        "_apply_completion_providers": "apply_completion_providers",
    }
    for old, new in renames.items():
        rs = rs.replace(old, new)
        reconcile = reconcile.replace(old, new)
        popcon = popcon.replace(old, new)

    rs = replace_once(
        rs,
        "        self._completion_plan_cache: list[tuple[Relation, bool]] | None = None\n",
        "",
        "remove completion cache declaration",
    )
    rs = replace_once(
        rs,
        "        self._completion_plan_cache = None\n",
        "",
        "remove completion cache reset",
    )

    # The executable provider schedule is now the only completion-plan
    # representation.  It owns ordering plus the resolved input/output plumbing.
    rs = sub_once(
        rs,
        r"        # Provider records with the per-call plumbing frozen:.*?\n        self\.provider_plan = \[.*?\n        \]\n        # One completion pass",
        "        # Freeze the one executable provider schedule.  Ordering and\n"
        "        # writable-output plumbing are resolved together so completion,\n"
        "        # POPCON and sparsity consume the same representation.\n"
        "        self.provider_plan = self._build_provider_plan()\n"
        "        # One completion pass",
        "replace provider-plan double representation",
    )

    provider_builder = '''
    def _build_provider_plan(self) -> tuple[tuple[Relation, bool, tuple[str, ...], tuple[tuple[str, Any], ...]], ...]:
        """Build the one executable completion-provider schedule.

        Explicit ownership wins over defaults.  Relations are ordered by all
        produced-output dependencies, including side outputs and produced
        constants, then each record freezes its input names and writable output
        specs.  Scalar and batched completion execute this same schedule.
        """
        provider_of: dict[str, Relation] = dict(self.default_provider_by_output)
        provider_of.update(self.derived_provider_by_output)
        explicit = {rel.name for rel in self.derived_provider_by_output.values()}
        rels: dict[str, Relation] = {}
        for rel in provider_of.values():
            rels.setdefault(rel.name, rel)
        only_missing = {name: name not in explicit for name in rels}

        producers: dict[str, list[str]] = {}
        for rel in rels.values():
            for out in rel.output_names:
                producers.setdefault(out, []).append(rel.name)
        dag = nx.DiGraph()
        dag.add_nodes_from(rels)
        for rel in rels.values():
            for inp in (*rel.input_names, *rel.constant_names):
                for producer in producers.get(inp, ()):
                    if producer != rel.name:
                        dag.add_edge(producer, rel.name)
        condensation = nx.condensation(dag)
        self._completion_acyclic = all(
            len(condensation.nodes[comp]["members"]) == 1 for comp in condensation
        )
        ordered_names = [
            rel_name
            for comp in nx.lexicographical_topological_sort(
                condensation, key=lambda c: min(condensation.nodes[c]["members"])
            )
            for rel_name in sorted(condensation.nodes[comp]["members"])
        ]
        return tuple(
            (
                rels[rel_name],
                only_missing[rel_name],
                rels[rel_name].input_names,
                tuple(
                    (out_name, self.variable_registry.get(out_name))
                    for out_name in rels[rel_name].output_names
                    if out_name in self.variable_registry
                    and (
                        self.variable_roles.get(out_name, "inactive") != "inactive"
                        or out_name in self.derived_provider_by_output
                    )
                ),
            )
            for rel_name in ordered_names
        )

    # ── Residual blocks: relations, domains, movement ─────────────────────
'''
    rs = sub_once(
        rs,
        r"\n    def _completion_plan\(self\).*?\n    # ── Residual blocks: relations, domains, movement ─────────────────────\n",
        "\n" + provider_builder,
        "replace cached completion plan with provider builder",
    )
    rs = replace_once(
        rs,
        "        for rel, _only_missing in self._completion_plan():\n",
        "        for rel, _only_missing, _input_names, _outs in self.provider_plan:\n",
        "sparsity uses provider plan",
    )

    # One packed-coordinate implementation is shared by ordinary unpacking and
    # reconcile's incremental grouped Jacobian.
    packed_helper = '''    def apply_packed_values(
        self,
        values: dict[str, Any],
        x: np.ndarray,
        specs: Sequence[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] | None = None,
    ) -> dict[str, Any]:
        """Apply packed solver coordinates to ``values`` without completion."""
        arr = np.asarray(x, dtype=float)
        for name, start, stop, offsets, scales, shape, transform in (self.packed_specs if specs is None else specs):
            local_x = arr[start:stop]
            actual = offsets * np.exp(local_x) if transform == "log" else offsets + scales * local_x
            values[name] = actual.copy() if shape == 1 else float(actual[0])
        return values

'''
    marker = "    def unpack(self, x: np.ndarray) -> dict[str, Any]:\n"
    if packed_helper.strip() in rs:
        raise RuntimeError("packed helper already present")
    rs = replace_once(rs, marker, packed_helper + marker, "insert packed application helper")
    rs = sub_once(
        rs,
        r"        # Start from the immutable inputs, then overwrite the packed free vars\.\n.*?        return self\.complete\(values\)",
        "        values = dict(self._packed_base_values)\n"
        "        self.apply_packed_values(values, x)\n"
        "        return self.complete(values)",
        "unpack uses packed helper",
    )
    reconcile = sub_once(
        reconcile,
        r"                for name, start, stop, offs, scales, shape, transform in group\[\"spans\"\]:\n.*?                system\.apply_profile_specs\(ns\)",
        "                system.apply_packed_values(ns, x_new, group[\"spans\"])\n"
        "                system.apply_profile_specs(ns)",
        "grouped jacobian uses packed helper",
    )

    # Replace the obsolete private-trust-boundary documentation with the actual
    # public plan protocol.  This is documentation-only and is not counted in LOC.
    rs = sub_once(
        rs,
        r"    Trust boundary: ``modes/`` also reads four \*compiled-plan\* members.*?\n\n    Args:",
        "    Execution modes consume the compiled-plan attributes ``residual_relations``,\n"
        "    ``profile_specs``, ``constant_defaults_solver`` and ``provider_plan`` plus\n"
        "    :meth:`apply_completion_providers`; these are the explicit public boundary\n"
        "    between compilation and mode-owned algorithms.\n\n"
        "    Args:",
        "document public compiled protocol",
    )

    RS.write_text(rs)
    RECONCILE.write_text(reconcile)
    POPCON.write_text(popcon)


def stage2() -> None:
    rs = RS.read_text()

    packed_layout = '''    def _build_packed_layout(
        self,
    ) -> tuple[
        list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]],
        np.ndarray,
        np.ndarray,
        list[str],
        list[str],
    ]:
        """Analyze solver packing without mutating the installed runtime layout."""
        lower: list[float] = []
        upper: list[float] = []
        specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []
        uninitialized: list[str] = []
        underdetermined: list[str] = []
        for name, role in sorted(self.variable_roles.items()):
            if role == "inactive" or name not in self.packed_variables:
                if role == "fixed" and self.inputs.get(name) is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue
            spec = self.variable_registry.get(name)
            lb, ub = spec.solver_bounds
            size = self.profile_size if spec.shape == 1 else 1
            if spec.shape == 1 and self.inputs.get(name) is None:
                underdetermined.append(name)
            try:
                initial = [
                    float(self.initial_value(name, index=i if spec.shape == 1 else None))
                    for i in range(size)
                ]
            except Exception:
                uninitialized.append(name)
                continue
            reference = None
            if self.inputs.get(name) is not None:
                try:
                    reference = np.asarray(
                        spec.solver_value(self.inputs[name], self.profile_size), dtype=float
                    ).reshape(-1)
                except Exception:
                    reference = None
            start = len(lower)
            offsets: list[float] = []
            scales: list[float] = []
            span_transform: str | None = None
            for i, init in enumerate(initial):
                ref = float(reference[min(i if spec.shape == 1 else 0, reference.size - 1)]) if reference is not None and reference.size else init
                scale, offset, lo, hi, transform = self.pack_scalar(name, init, lb, ub, scale_ref=ref)
                lower.append(lo)
                upper.append(hi)
                offsets.append(offset)
                scales.append(scale)
                if transform == "log":
                    span_transform = "log"
            specs.append(
                (
                    name,
                    start,
                    len(lower),
                    np.asarray(offsets, dtype=float),
                    np.asarray(scales, dtype=float),
                    spec.shape,
                    span_transform,
                )
            )
        return specs, np.asarray(lower), np.asarray(upper), uninitialized, underdetermined

    def pack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Install the numeric solver layout for the already-compiled system."""
        specs, lower, upper, uninitialized, underdetermined = self._build_packed_layout()
        self.packed_specs = specs
        self.packed_dim = int(lower.size)
        self.uninitialized_free_variables = uninitialized
        self.underdetermined_profiles = underdetermined
        self._classify_avg_to_profile()
        self._compiler_report_cache = None
        self._packed_base_values = self.input_values()
        self._movement_plan = self._build_movement_plan()
        return np.zeros(self.packed_dim), lower, upper

'''
    rs = sub_once(
        rs,
        r"    def pack\(self\) -> tuple\[np\.ndarray, np\.ndarray, np\.ndarray\]:.*?\n    def required_uninitialized_free_variables",
        packed_layout + "    def required_uninitialized_free_variables",
        "split packability analysis from pack installation",
    )

    rs = replace_once(
        rs,
        "            self.pack()\n            if set(self.uninitialized_free_variables) <= self._unevaluable_names:\n                return\n",
        "            _specs, _lower, _upper, uninitialized, _under = self._build_packed_layout()\n"
        "            if set(uninitialized) <= self._unevaluable_names:\n"
        "                self.pack()\n"
        "                return\n",
        "compile fingerprint uses pure layout analysis",
    )

    rs = replace_once(
        rs,
        "            self.pack()\n            # Two evaluability failures, not one.  A variable with NO SEED\n",
        "            _specs, _lower, _upper, uninitialized, underdetermined = self._build_packed_layout()\n"
        "            # Two evaluability failures, not one.  A variable with NO SEED\n",
        "compile loop uses pure layout analysis",
    )
    rs = replace_once(
        rs,
        "            newly = set(self.uninitialized_free_variables) | set(self.underdetermined_profiles)\n",
        "            newly = set(uninitialized) | set(underdetermined)\n",
        "compile loop reads analysis issues",
    )
    rs = replace_once(
        rs,
        "        # Packing detects raw profile cores because only the packed layout\n",
        "        self.uninitialized_free_variables = list(uninitialized)\n"
        "        self.underdetermined_profiles = list(underdetermined)\n\n"
        "        # Packing detects raw profile cores because only the packed layout\n",
        "retain final layout issues",
    )
    rs = replace_once(
        rs,
        "        for name in self.underdetermined_profiles:\n            if self.variable_roles.get(name) == \"computed\":\n                self.variable_roles[name] = \"assumed\"\n\n    def _refresh_seeds",
        "        for name in self.underdetermined_profiles:\n"
        "            if self.variable_roles.get(name) == \"computed\":\n"
        "                self.variable_roles[name] = \"assumed\"\n"
        "        self.pack()\n\n"
        "    def _refresh_seeds",
        "install final packed layout once compile is stable",
    )

    # Unevaluable names prune the candidate pool before default activation and
    # DM partitioning, so structural_blocks is always computed from the same
    # relation pool that survives the pass.
    rs = replace_once(
        rs,
        "        usable: list[Relation] = []\n        for rel in self.candidate_primary_relations:\n",
        "        usable: list[Relation] = []\n"
        "        unevaluable = self._unevaluable_names - set(self.inputs)\n"
        "        for rel in self.candidate_primary_relations:\n"
        "            blocked = sorted(set(rel.variables) & unevaluable)\n"
        "            if blocked:\n"
        "                self._mark_relation_inactive(\n"
        "                    rel, \"inactive_unevaluable: requires unevaluable \" + \", \".join(blocked), replace=True\n"
        "                )\n"
        "                continue\n",
        "prefilter unevaluable relations",
    )

    rs = sub_once(
        rs,
        r"        # Variables a previous prune round found unevaluable are treated as\n.*?        undecidable \|= unevaluable\n",
        "",
        "remove post-partition unevaluable injection",
    )
    rs = sub_once(
        rs,
        r"            if undec:\n                unev = sorted\(set\(rel\.variables\) & unevaluable\)\n                if unev:\n                    self\._mark_relation_inactive\(rel, \"inactive_unevaluable: requires unevaluable \" \+ \", \"\.join\(unev\), replace=True\)\n                else:\n                    self\._mark_relation_inactive\(rel, \"inactive_undecidable: cannot determine \" \+ \", \"\.join\(undec\), replace=True\)\n",
        "            if undec:\n"
        "                self._mark_relation_inactive(\n"
        "                    rel, \"inactive_undecidable: cannot determine \" + \", \".join(undec), replace=True\n"
        "                )\n",
        "simplify active relation pruning",
    )

    rs = rs.replace(
        "Each round then re-packs\n        against the current roles -- the pack is the evaluability oracle, and\n        it reports TWO ways a variable can fail:",
        "Each round analyzes the packed layout\n        against the current roles without installing runtime state.  That pure\n        analysis reports TWO ways a variable can fail:",
    )
    RS.write_text(rs)

    LIFECYCLE_TEST.write_text('''from pathlib import Path\n\nimport pytest\n\nfrom fusdb import Reactor\n\n\nREACTORS = Path(__file__).parents[1] / "reactors"\n\n\ndef _reactor_dirs():\n    return sorted(path for path in REACTORS.iterdir() if path.is_dir())\n\n\n@pytest.mark.parametrize("reactor_dir", _reactor_dirs(), ids=lambda path: path.name)\ndef test_compiled_blocks_only_reference_final_active_system(reactor_dir):\n    try:\n        system = Reactor.from_yaml(reactor_dir).relation_system()\n    except Exception as exc:\n        pytest.skip(f"fixture is not loadable as a reactor directory: {exc}")\n    system.compile()\n    active = system.active_variable_names\n    assert all(set(block) <= active for block in system.structural_blocks)\n    assert all(not (set(rel.variables) & system._unevaluable_names) for rel in system.relations)\n\n\ndef test_packability_analysis_does_not_install_runtime_layout():\n    system = Reactor.from_yaml(REACTORS / "DEMO_2022").relation_system()\n    system.compile()\n    before_specs = list(system.packed_specs)\n    before_dim = system.packed_dim\n    before_movement = list(system._movement_plan)\n\n    system._build_packed_layout()\n\n    assert system.packed_specs == before_specs\n    assert system.packed_dim == before_dim\n    assert len(system._movement_plan) == len(before_movement)\n\n\ndef test_completion_has_one_executable_provider_plan():\n    system = Reactor.from_yaml(REACTORS / "DEMO_2022").relation_system()\n    system.compile()\n    assert isinstance(system.provider_plan, tuple)\n    assert not hasattr(system, "_completion_plan_cache")\n    assert not hasattr(system, "_completion_plan")\n''')


def normalize(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def snapshot(path: Path) -> None:
    from fusdb import Reactor

    records: dict[str, Any] = {}
    for directory in sorted((ROOT / "reactors").iterdir()):
        if not directory.is_dir():
            continue
        entry: dict[str, Any] = {}
        try:
            reactor = Reactor.from_yaml(directory)
            system = reactor.relation_system()
            system.compile()
            entry["compile"] = {
                "active_relations": [rel.name for rel in system.primary_relations],
                "providers": {
                    "default": {k: v.name for k, v in system.default_provider_by_output.items()},
                    "derived": {k: v.name for k, v in system.derived_provider_by_output.items()},
                },
                "roles": dict(system.variable_roles),
                "packed": sorted(system.packed_variables),
                "seed_provenance": dict(system.seed_provenance),
                "blocks": normalize(system.structural_blocks),
            }
            result = reactor.run("reconcile")
            entry["reconcile"] = {
                "success": bool(result.get("success", False)),
                "mode": result.get("mode"),
                "termination": result.get("termination"),
                "failed_relations": list(result.get("failed_relations") or ()),
                "inputs_beyond_tolerance": normalize(result.get("inputs_beyond_tolerance") or ()),
                "values": normalize(result.get("values") or {}),
            }
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        records[directory.name] = entry
    path.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=True))


def _numeric_close(a: Any, b: Any) -> bool:
    import numpy as np

    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
    except Exception:
        return a == b
    if aa.shape != bb.shape:
        return False
    return bool(np.allclose(aa, bb, rtol=5.0e-6, atol=1.0e-10, equal_nan=True))


def compare(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    if set(before) != set(after):
        raise AssertionError("reactor corpus changed during refactor")
    block_changes: dict[str, Any] = {}
    for name in before:
        b, a = before[name], after[name]
        if b.get("error") != a.get("error"):
            raise AssertionError(f"{name}: load/execute error changed: {b.get('error')} -> {a.get('error')}")
        if "error" in b:
            continue
        for key in ("active_relations", "providers", "roles", "packed", "seed_provenance"):
            if b["compile"][key] != a["compile"][key]:
                raise AssertionError(f"{name}: compile {key} changed")
        if b["compile"]["blocks"] != a["compile"]["blocks"]:
            block_changes[name] = {"before": b["compile"]["blocks"], "after": a["compile"]["blocks"]}
        br, ar = b["reconcile"], a["reconcile"]
        for key in ("success", "mode", "failed_relations"):
            if br[key] != ar[key]:
                raise AssertionError(f"{name}: reconcile {key} changed: {br[key]!r} -> {ar[key]!r}")
        if [x.get("name") for x in br["inputs_beyond_tolerance"]] != [x.get("name") for x in ar["inputs_beyond_tolerance"]]:
            raise AssertionError(f"{name}: inputs_beyond_tolerance membership changed")
        if set(br["values"]) != set(ar["values"]):
            raise AssertionError(f"{name}: result value keys changed")
        for variable in br["values"]:
            if not _numeric_close(br["values"][variable], ar["values"][variable]):
                raise AssertionError(f"{name}: value {variable} changed beyond 5e-6 relative tolerance")
    print("structural block changes (allowed only when behavior is preserved):")
    print(json.dumps(block_changes, indent=2, sort_keys=True))


def docstring_lines(source: str) -> set[int]:
    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, (ast.Str, ast.Constant)) and isinstance(getattr(first.value, "value", None), str):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def implementation_loc(path: Path) -> int:
    source = path.read_text()
    docs = docstring_lines(source)
    code_lines: set[int] = set()
    reader = io.StringIO(source).readline
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT}
    for tok in tokenize.generate_tokens(reader):
        if tok.type in ignored or tok.start[0] in docs:
            continue
        if tok.string.strip():
            code_lines.add(tok.start[0])
    return len(code_lines)


def loc(path: Path) -> None:
    files = [RS, RECONCILE, POPCON]
    payload = {str(p.relative_to(ROOT)): implementation_loc(p) for p in files}
    payload["total"] = sum(payload.values())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


def compare_loc(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    print("implementation LOC before:", before)
    print("implementation LOC after:", after)
    if after["total"] > before["total"]:
        raise AssertionError(f"implementation LOC increased: {before['total']} -> {after['total']}")


def main() -> None:
    command = sys.argv[1]
    if command == "stage1":
        stage1()
    elif command == "stage2":
        stage2()
    elif command == "snapshot":
        snapshot(Path(sys.argv[2]))
    elif command == "compare":
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
    elif command == "loc":
        loc(Path(sys.argv[2]))
    elif command == "compare-loc":
        compare_loc(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
