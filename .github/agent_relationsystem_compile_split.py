from pathlib import Path

path = Path('src/fusdb/relationsystem.py')
text = path.read_text()

start = text.index('    def _run_compile_pass(self) -> None:')
freeze_marker = text.index('        # ── Refresh variable scales and tolerances used by residuals.', start)
end = text.index('\n\n    def _usable_candidates', freeze_marker)
old_block = text[start:end]
struct_part = text[start:freeze_marker]
freeze_body = text[freeze_marker:end]

struct_part = struct_part.replace(
    '    def _run_compile_pass(self) -> None:\n        """Run one full structural compile pass over the current candidates.\n',
    '    def _compile_structure_pass(self) -> None:\n        """Compile only scenario structure over the prepared model.\n',
    1,
)
# Replace the stale long description with a precise structural-only one.
doc_start = struct_part.index('        """Compile only scenario structure')
doc_end = struct_part.index('        """', doc_start + 12) + len('        """')
struct_part = struct_part[:doc_start] + '''        """Compile active relations, determinacy, providers, blocks and roles.

        This pass deliberately creates no numerical execution products. Pruning
        may repeat it until the active structure reaches a fixed point; only the
        final structure is frozen for execution by :meth:`_freeze_execution_plan`.
        """''' + struct_part[doc_end:]

freeze_method = '''    def _freeze_execution_plan(self) -> None:\n        """Build numerical execution products once for the final structure."""\n''' + freeze_body
text = text[:start] + struct_part + '\n\n' + freeze_method + text[end:]

text = text.replace('            self._run_compile_pass()\n', '            self._compile_structure_pass()\n')
# Freeze only after pruning has reached its final structure and before pack.
needle = '''        for name in self.underdetermined_profiles:\n            if self.variable_roles.get(name) == "computed":\n                self.variable_roles[name] = "assumed"\n        self.pack()\n'''
replacement = '''        for name in self.underdetermined_profiles:\n            if self.variable_roles.get(name) == "computed":\n                self.variable_roles[name] = "assumed"\n        self._freeze_execution_plan()\n        self.pack()\n'''
if needle not in text:
    raise SystemExit('compile finalization block not found')
text = text.replace(needle, replacement, 1)

# Cache invalidation belongs to structural passes; execution freeze has no reason
# to invalidate the compiler report again.
text = text.replace('''        # ── Reset the graph verdicts and the caches derived from the previous\n''', '''        # ── Reset the graph verdicts and the caches derived from the previous\n''')

path.write_text(text)
print('split structural compilation from final execution-plan freeze')
