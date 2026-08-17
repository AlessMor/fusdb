from pathlib import Path
import subprocess

# Rebuild POPCON from the validated pre-refactor implementation, changing only
# worker model reconstruction and the now-private plan recompile hook. This
# preserves the existing warm-start/value-state behavior that selects the
# established nonlinear solution branch.
original = subprocess.check_output(
    ['git', 'show', 'HEAD:src/fusdb/modes/popcon.py'], text=True
)
text = original
if 'import pickle\n' not in text:
    text = text.replace('from functools import partial\n', 'from functools import partial\nimport pickle\n')

needle = '\ndef _solve_batched_cases_from_spec(\n'
pos = text.index(needle)
worker_cache = '''\n_WORKER_MODELS: dict[bytes, Any] = {}\n\n\ndef _worker_model(spec: Mapping[str, Any]) -> Any:\n    """Return the process-local prepared model for one picklable worker spec."""\n    key = pickle.dumps(spec, protocol=5)\n    model = _WORKER_MODELS.get(key)\n    if model is None:\n        model = _rebuild_system(spec)\n        _WORKER_MODELS[key] = model\n    return model\n\n'''
text = text[:pos] + worker_cache + text[pos:]
text = text.replace('plan = _rebuild_system(spec).compile()', 'plan = _worker_model(spec).compile()')
# CompilePlan is compiled on construction, but POPCON intentionally mutates its
# own ephemeral scan state to retain its historical warm start. Keep this as a
# private algorithm hook, not a public plan lifecycle.
text = text.replace('self.compile()', 'self._recompile()')
Path('src/fusdb/modes/popcon.py').write_text(text)

rs = Path('src/fusdb/relationsystem.py')
rtext = rs.read_text()
if '    def _recompile(self) -> "CompilePlan":' not in rtext:
    marker = '    def run(self, mode: str = "verify"'
    idx = rtext.index(marker, rtext.index('class CompilePlan:'))
    hook = '''    def _recompile(self) -> "CompilePlan":\n        """Recompile this ephemeral plan after an internal algorithm mutates scenario state."""\n        self._compile_with_pruning()\n        return self\n\n'''
    rtext = rtext[:idx] + hook + rtext[idx:]
rs.write_text(rtext)

# Preserve the worker-reuse contract without asserting that public POPCON uses a
# fresh plan (it deliberately keeps the validated warm-start behavior).
test = Path('tests/test_popcon_source_profiles.py')
t = test.read_text()
if 'def test_worker_recipe_reuses_prepared_model_in_process()' not in t:
    t += '''\n\ndef test_worker_recipe_reuses_prepared_model_in_process():\n    plan = _source_system(profile_size=31)\n    spec = popcon_mode._system_spec(plan)\n    popcon_mode._WORKER_MODELS.clear()\n    first = popcon_mode._worker_model(spec)\n    second = popcon_mode._worker_model(spec)\n    assert first is second\n    first_plan = first.compile()\n    second_plan = second.compile()\n    assert second_plan._structure_cache_hit\n'''
test.write_text(t)
print('preserved POPCON warm-start semantics while reusing worker RelationSystem models')
