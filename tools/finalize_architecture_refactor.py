from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    file = ROOT / path
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}\n--- pattern ---\n{old}")
    file.write_text(text.replace(old, new, 1))


replace_once(
    "src/fusdb/relationsystem.py",
    "from .relation import COORDINATE_NAMES, Relation, canonicalize_relation, canonicalize_relation_names, constraint_from_expression, is_default_relation\n",
    "from .relation import COORDINATE_NAMES, Relation, build_constraint_relations, canonicalize_relation, canonicalize_relation_names, is_default_relation\n",
)
replace_once(
    "src/fusdb/relationsystem.py",
    "from .utils import ZERO_TOL, parse_constraint_specs, signed_scalar_grid, value_in_domain, volume_average\n",
    "from .utils import ZERO_TOL, signed_scalar_grid, value_in_domain, volume_average\n",
)
replace_once(
    "src/fusdb/relationsystem.py",
    '''        self.system_constraint_relations = tuple(\n            canonicalize_relation_names(\n                constraint_from_expression(\n                    text,\n                    name=f"system_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="system",\n                    source_name=self.name,\n                ),\n                self.variable_registry,\n            )\n            for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))\n        )\n''',
    '''        self.system_constraint_relations = tuple(\n            canonicalize_relation_names(guard, self.variable_registry)\n            for guard in build_constraint_relations(\n                constraints,\n                name_prefix="system_constraint",\n                source_kind="system",\n                source_name=self.name,\n            )\n        )\n''',
)
