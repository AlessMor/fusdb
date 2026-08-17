from pathlib import Path

# CompilePlan objects are compiled on construction; remove stale second compile
# calls from tests that already obtain a plan via RelationSystem.compile().
for path in Path('tests').rglob('*.py'):
    text = path.read_text()
    original = text
    # Common migrated patterns: an assignment ending in .compile() followed by
    # an immediate compile() on the assigned plan.
    lines = text.splitlines()
    out = []
    compiled_vars = set()
    for line in lines:
        stripped = line.strip()
        if '=' in stripped and stripped.endswith('.compile()'):
            lhs = stripped.split('=', 1)[0].strip()
            if lhs.isidentifier():
                compiled_vars.add(lhs)
        if stripped.endswith('.compile()') and stripped[:-10].isidentifier():
            name = stripped[:-10]
            if name in compiled_vars:
                continue
        out.append(line)
    text = '\n'.join(out) + ('\n' if original.endswith('\n') else '')
    if text != original:
        path.write_text(text)

print('removed redundant CompilePlan.compile() call sites')
