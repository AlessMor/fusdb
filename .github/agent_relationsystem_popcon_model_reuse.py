from pathlib import Path

pop = Path('src/fusdb/modes/popcon.py')
text = pop.read_text()
if 'import pickle\n' not in text:
    text = text.replace('from functools import partial\n', 'from functools import partial\nimport pickle\n')

# Reuse one reconstructed RelationSystem per worker process/spec so repeated
# chunks share canonical preparation and the model-owned structural cache.
needle = '\ndef _solve_batched_cases_from_spec(\n'
pos = text.index(needle)
worker_cache = '''\n_WORKER_MODELS: dict[bytes, Any] = {}\n\n\ndef _worker_model(spec: Mapping[str, Any]) -> Any:\n    key = pickle.dumps(spec, protocol=5)\n    model = _WORKER_MODELS.get(key)\n    if model is None:\n        model = _rebuild_system(spec)\n        _WORKER_MODELS[key] = model\n    return model\n\n'''
text = text[:pos] + worker_cache + text[pos:]
text = text.replace('''    plan = _rebuild_system(spec).compile()\n''', '''    plan = _worker_model(spec).compile()\n''')

# POPCON no longer mutates/recompiles the caller's plan. Build a new plan for
# the pinned midpoint scenario from the same reusable model, then scan with it.
try_start = text.index('    try:\n        # Pin the axes to the user-requested grid coordinates', text.index('def run('))
finally_start = text.index('    finally:\n', try_start)
final_end = text.index('\n\n    n_failed =', finally_start)
try_body = text[try_start + len('    try:\n'):finally_start]
body_marker = try_body.index('        if outputs is None:')
body_rest = try_body[body_marker:]
body_rest = '\n'.join(line[4:] if line.startswith('    ') else line for line in body_rest.split('\n'))
scan_setup = '''    # Compile a fresh scan plan instead of mutating the caller's plan. Passing\n    # explicit None for model-base inputs absent from this scenario preserves\n    # the complete supplied/missing state. The model itself is never changed.\n    scenario_names = set(self.model.base_inputs) | set(self.inputs)\n    scenario_inputs = {name: self.inputs.get(name) for name in scenario_names}\n    for name, values in ((x_name, x_values), (y_name, y_values)):\n        scenario_inputs[name] = float(values[values.size // 2])\n    scan_fixed = set(self.fixed) | {x_name, y_name}\n    self = self.model.compile(inputs=scenario_inputs, fixed=scan_fixed)\n\n'''
text = text[:try_start] + scan_setup + body_rest.rstrip() + text[final_end:]

# Remove the now-unused saved-state setup.
saved_start = text.find('    # Full pre-scan state, restored verbatim when the scan finishes.\n', text.index('def run('))
if saved_start >= 0:
    nx_pos = text.index('    nx, ny =', saved_start)
    text = text[:saved_start] + text[nx_pos:]

pop.write_text(text)

# The temporary internal recompile hook is no longer needed anywhere.
rs = Path('src/fusdb/relationsystem.py')
rtext = rs.read_text()
block = '''    def _recompile(self) -> "CompilePlan":\n        """Internal compatibility hook for algorithms that intentionally mutate scenario state."""\n        self._compile_with_pruning()\n        return self\n\n'''
rtext = rtext.replace(block, '')
rs.write_text(rtext)

# Add focused process-local model reuse contract.
test = Path('tests/test_popcon_source_profiles.py')
t = test.read_text()
t += '''\n\ndef test_worker_recipe_reuses_prepared_model_in_process():\n    plan = _source_system(profile_size=31)\n    spec = popcon_mode._system_spec(plan)\n    popcon_mode._WORKER_MODELS.clear()\n    first = popcon_mode._worker_model(spec)\n    second = popcon_mode._worker_model(spec)\n    assert first is second\n    first_plan = first.compile()\n    second_plan = second.compile()\n    assert second_plan._structure_cache_hit\n'''
test.write_text(t)
print('migrated POPCON to fresh scan plans and per-worker model reuse')
