from pathlib import Path

relation = Path('src/fusdb/relation.py')
r = relation.read_text()
anchor = 'class RelationNotInvertibleError(RelationSolveError):\n'
insert = '''def build_constraint_relations(constraints, *, name_prefix: str, source_kind: str, source_name: str) -> tuple["Relation", ...]:\n    """Normalize constraint specs into relation guards in one place."""\n    return tuple(\n        constraint_from_expression(\n            text,\n            name=f"{name_prefix}_{index}",\n            enforce=enforce,\n            source_kind=source_kind,\n            source_name=source_name,\n        )\n        for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))\n    )\n\n\n'''
if 'def build_constraint_relations(' not in r:
    r = r.replace(anchor, insert + anchor)
old = '''        # Local constraints are themselves relations. enforce=False means checked-only applicability.\n        built: list[Relation] = []\n        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):\n            built.append(\n                constraint_from_expression(\n                    text,\n                    name=f"{self.name}_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="relation",\n                    source_name=self.name,\n                )\n            )\n        self.constraint_relations = tuple(built)\n'''
new = '''        # Local constraints are themselves relations. enforce=False means checked-only applicability.\n        self.constraint_relations = build_constraint_relations(\n            self.constraints,\n            name_prefix=f"{self.name}_constraint",\n            source_kind="relation",\n            source_name=self.name,\n        )\n'''
if old not in r:
    raise SystemExit('relation constraint block not found')
r = r.replace(old, new)
relation.write_text(r)

variable = Path('src/fusdb/variable.py')
v = variable.read_text()
v = v.replace('from .relation import Relation, constraint_from_expression', 'from .relation import Relation, build_constraint_relations')
v = v.replace('from .utils import coerce_numeric_value, coerce_to_shape, parse_constraint_specs, unique_preserve_order, value_in_domain', 'from .utils import coerce_numeric_value, coerce_to_shape, unique_preserve_order, value_in_domain')
old = '''        built: list[Relation] = []\n        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):\n            built.append(\n                constraint_from_expression(\n                    text,\n                    name=f"{self.name}_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="variable",\n                    source_name=self.name,\n                )\n            )\n        object.__setattr__(self, "relations", tuple(built))\n'''
new = '''        object.__setattr__(\n            self,\n            "relations",\n            build_constraint_relations(\n                self.constraints,\n                name_prefix=f"{self.name}_constraint",\n                source_kind="variable",\n                source_name=self.name,\n            ),\n        )\n'''
if old not in v:
    raise SystemExit('variable constraint block not found')
v = v.replace(old, new)
variable.write_text(v)

registry = Path('src/fusdb/registry/variable_registry.py')
t = registry.read_text()
old = '''        cached = _SPEC_GUARDS.get(self.name)\n        if cached is None:\n            from ..relation import constraint_from_expression\n\n            cached = tuple(\n                constraint_from_expression(\n                    text,\n                    name=f"{self.name}_registry_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="variable",\n                    source_name=self.name,\n                )\n                for index, (text, enforce) in enumerate(self.constraints)\n            )\n            _SPEC_GUARDS[self.name] = cached\n        return cached\n'''
new = '''        cached = _SPEC_GUARDS.get(self.name)\n        if cached is None:\n            from ..relation import build_constraint_relations\n\n            cached = build_constraint_relations(\n                self.constraints,\n                name_prefix=f"{self.name}_registry_constraint",\n                source_kind="variable",\n                source_name=self.name,\n            )\n            _SPEC_GUARDS[self.name] = cached\n        return cached\n'''
if old not in t:
    raise SystemExit('registry constraint block not found')
t = t.replace(old, new)
registry.write_text(t)
