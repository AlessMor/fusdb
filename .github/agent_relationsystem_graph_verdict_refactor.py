from pathlib import Path

path = Path('src/fusdb/relationsystem.py')
text = path.read_text()

# Plan-local verdict state is stored as ordinary maps/sets, never on a copied graph.
text = text.replace(
'''        self._graph: nx.DiGraph | None = None
        self._provider_view: tuple[dict[str, Relation], dict[str, Relation]] | None = None
        self._dependency_closure_cache: tuple[
''',
'''        self._active_relation_names: set[str] = set()
        self._inactive_reasons: dict[str, str] = {}
        self._decidability: dict[str, str] = {}
        self._default_providers: dict[str, Relation] = {}
        self._derived_providers: dict[str, Relation] = {}
        self._dependency_closure_cache: tuple[
''')

start = text.index('    def _structural_graph(self) -> nx.DiGraph:', text.index('class CompilePlan:'))
end = text.index('    @staticmethod\n    def _exclusive_unknowns', start)
replacement = '''    def _structural_graph(self) -> nx.DiGraph:
        """Return the model's canonical immutable bipartite topology."""
        return self.model.graph

    # ── Per-plan structural verdicts ─────────────────────────────────────

    def _reset_graph_verdicts(self) -> None:
        """Reset scenario-specific compile verdicts without touching topology."""
        self._active_relation_names.clear()
        self._inactive_reasons.clear()
        self._decidability.clear()
        self._default_providers.clear()
        self._derived_providers.clear()

'''
text = text[:start] + replacement + text[end:]

start = text.index('    def _mark_relation_active', text.index('class CompilePlan:'))
end = text.index('    def _classify_avg_to_profile', start)
replacement = '''    def _mark_relation_active(self, rel: Relation) -> None:
        self._active_relation_names.add(rel.name)
        self._inactive_reasons.pop(rel.name, None)

    def _mark_relation_inactive(self, rel: Relation, reason: str, *, replace: bool = False) -> None:
        """Record one relation's inactivation reason (first reason wins unless replaced)."""
        self._active_relation_names.discard(rel.name)
        if replace or rel.name not in self._inactive_reasons:
            self._inactive_reasons[rel.name] = reason

    def _set_provider_edge(self, name: str, rel: Relation, kind: str) -> None:
        """Select the plan-local provider of ``name`` without mutating the graph."""
        target = self._default_providers if kind == "default" else self._derived_providers
        target[name] = rel

    @property
    def default_provider_by_output(self) -> dict[str, Relation]:
        return self._default_providers

    @property
    def derived_provider_by_output(self) -> dict[str, Relation]:
        return self._derived_providers

    @property
    def blocked_relation_reasons(self) -> dict[str, str]:
        return self._inactive_reasons

'''
text = text[:start] + replacement + text[end:]

# Decidability is a compile verdict, not graph metadata.
old = '''        graph = self._structural_graph()
        for node, data in graph.nodes(data=True):
            if data["kind"] != "variable":
                continue
            name = node[1]
            if name in supplied:
                data["decidability"] = "supplied"
            elif name in known_defaults:
                data["decidability"] = "default"
            elif name in strict_forward:
                data["decidability"] = "forward"
            elif name in forward:
                data["decidability"] = "acausal"
'''
new = '''        for name in self.known:
            if name in supplied:
                self._decidability[name] = "supplied"
            elif name in known_defaults:
                self._decidability[name] = "default"
            elif name in strict_forward:
                self._decidability[name] = "forward"
            elif name in forward:
                self._decidability[name] = "acausal"
'''
if old not in text:
    raise SystemExit('activate-default decidability block not found')
text = text.replace(old, new)

old = '''        graph = self._structural_graph()
        for name in block_decidable:
            node = ("variable", name)
            if node in graph and graph.nodes[node].get("decidability") is None:
                graph.nodes[node]["decidability"] = "block"
        for name in undecidable:
            node = ("variable", name)
            if node in graph:
                graph.nodes[node]["decidability"] = "underdetermined"
'''
new = '''        for name in block_decidable:
            self._decidability.setdefault(name, "block")
        for name in undecidable:
            self._decidability[name] = "underdetermined"
'''
if old not in text:
    raise SystemExit('partition decidability block not found')
text = text.replace(old, new)

old = '''            "decidability": dict(sorted(
                (node[1], data["decidability"])
                for node, data in self._structural_graph().nodes(data=True)
                if data.get("kind") == "variable" and data.get("decidability")
            )),
'''
new = '''            "decidability": dict(sorted(self._decidability.items())),
'''
if old not in text:
    raise SystemExit('compiler report decidability block not found')
text = text.replace(old, new)

# Comments/docstrings must no longer describe graph annotations as mutable state.
text = text.replace('''        # ── Reset the graph verdicts and the caches derived from the previous
        # provider/active-relation plan.  The graph's structure is immutable
        # (one node per candidate relation/variable); each compile pass
        # rewrites only the verdict annotations -- relation activation and
        # reason, provider edges, per-variable supplied/fixed flags.
''', '''        # Reset plan-local verdicts and caches derived from the previous
        # provider/active-relation plan. The model graph is never mutated.
''')
text = text.replace('''        Deactivates relations touching undecidable (or previously unevaluable)
        variables (graph verdicts), writes the active relation set, the
''', '''        Deactivates relations touching undecidable (or previously unevaluable)
        variables, writes the active relation set, the
''')
text = text.replace('''    Relation nodes are annotated with the ``relation`` object, its ordered
    ``variables`` tuple, ``enforce`` and ``is_default``; variable nodes carry
    only ``kind`` -- callers layer their own annotations on top (the compiled
    system adds ``shape`` plus the per-pass verdicts, the plotting views add
    display labels).
''', '''    Relation nodes are annotated with the ``relation`` object, its ordered
    ``variables`` tuple, ``enforce`` and ``is_default``; variable nodes carry
    structural metadata only. Scenario-specific compilation verdicts are kept
    on :class:`CompilePlan`, never written onto this topology.
''')

path.write_text(text)

# Strengthen contract: no plan graph allocation and model topology is unmodified.
test = Path('tests/test_relation_system_compile_plan.py')
t = test.read_text()
t = t.replace('''    assert plan.candidate_primary_relations == list(model.candidate_primary_relations)\n    assert plan.profile_size == model.profile_size\n''', '''    assert plan.candidate_primary_relations == list(model.candidate_primary_relations)\n    assert plan.profile_size == model.profile_size\n    assert not hasattr(plan, "_graph")\n    assert plan._structural_graph() is model.graph\n''')
t += '''\n\ndef test_compile_verdicts_do_not_annotate_model_graph():\n    model = _model()\n    plan = model.compile(fixed={"R"})\n    relation_node = ("relation", "Aspect ratio")\n    variable_node = ("variable", "R")\n    assert "active" not in model.graph.nodes[relation_node]\n    assert "inactive_reason" not in model.graph.nodes[relation_node]\n    assert "decidability" not in model.graph.nodes[variable_node]\n    assert all("provider" not in data for *_edge, data in model.graph.edges(data=True))\n    assert plan.compiler_report["decidability"]\n'''
test.write_text(t)
print('moved CompilePlan verdicts off the canonical graph')
