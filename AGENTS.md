# Code change rules

- Do not preserve backward compatibility or legacy code paths unless the task explicitly requires it.
- Prefer removing deprecated adapters, shims, feature flags, compatibility branches, and migration-only code when they are no longer needed.
- When simplifying code, choose the cleanest current design over transitional support.
- If a change would normally keep legacy behavior, stop and explain why that compatibility is necessary.
- Try to make code concise and maintainable
- avoid thin wrappers or helper called only once or twice, inline instead
- add brief comments to each logical step in functions/classes
- functions should have Google Style description
- run tests in conda .venv
- avoid to define __all__ or similar lists of functions/objects to export