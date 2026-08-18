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


def append_once(path: str, marker: str, addition: str) -> None:
    file = ROOT / path
    text = file.read_text()
    if marker in text:
        raise RuntimeError(f"{path}: marker already present: {marker}")
    file.write_text(text.rstrip() + "\n\n\n" + addition.rstrip() + "\n")


# Phase 3: generated source relations are already directly picklable. Remove the
# parallel rebuild representation and carry the Relation itself to POPCON workers.
replace_once(
    "src/fusdb/profile_sources.py",
    '''        source_kind="source_profile",\n        source_name=name,\n        rebuild_spec={\n            "kind": "source_profile",\n            "version": 1,\n            "variable": name,\n            "coordinate": coordinate,\n            "source_coordinate": source.copy(),\n            "source_values": source_values.copy(),\n            "fixed": bool(fixed),\n            "average_name": average_name,\n        },\n''',
    '''        source_kind="source_profile",\n        source_name=name,\n''',
)
replace_once(
    "src/fusdb/profile_sources.py",
    '''\ndef source_profile_relation_from_spec(spec: Mapping[str, Any]) -> Relation:\n    """Rebuild a generated source-profile relation from its worker recipe."""\n    if spec.get("kind") != "source_profile":\n        raise ValueError(f"Unsupported generated relation kind {spec.get('kind')!r}.")\n    if int(spec.get("version", 1)) != 1:\n        raise ValueError(f"Unsupported source-profile rebuild spec version {spec.get('version')!r}.")\n    average_name = spec.get("average_name")\n    return _source_profile_relation_from_data(\n        name=str(spec["variable"]),\n        coordinate=str(spec.get("coordinate") or "rho"),\n        source=np.asarray(spec["source_coordinate"], dtype=float),\n        source_values=np.asarray(spec["source_values"], dtype=float),\n        fixed=bool(spec.get("fixed", False)),\n        average_name=None if average_name is None else str(average_name),\n    )\n\n''',
    "\n",
)

replace_once(
    "src/fusdb/modes/popcon.py",
    '''def _relation_rebuild_spec(rel: Any) -> dict[str, Any]:\n    """Return the picklable worker recipe for one candidate relation."""\n    if rel.rebuild_spec is not None:\n        return dict(rel.rebuild_spec)\n    return {"kind": "registry", "name": rel.name}\n\n\ndef _system_spec(system: Any) -> dict[str, Any]:\n    """Picklable recipe to rebuild an equivalent system in a worker process.\n\n    Registry relations are represented by name. Runtime-generated relations\n    provide an explicit ``rebuild_spec`` containing only picklable immutable\n    source data. The common profile-grid size is carried separately so a source\n    profile sampled on an arbitrary external grid cannot reset worker geometry.\n    """\n''',
    '''def _relation_worker_item(rel: Any) -> Any:\n    """Return the minimal picklable worker representation for one relation."""\n    return rel if rel.source_kind == "source_profile" else rel.name\n\n\ndef _system_spec(system: Any) -> dict[str, Any]:\n    """Picklable recipe to rebuild an equivalent system in a worker process.\n\n    Registry relations are represented by name. Runtime-generated source-profile\n    relations are already picklable and are carried directly, so there is no\n    second reconstruction schema to keep in sync. The common profile-grid size\n    is carried separately so an arbitrary source sampling cannot reset worker\n    geometry.\n    """\n''',
)
replace_once(
    "src/fusdb/modes/popcon.py",
    '''        "relations": [_relation_rebuild_spec(rel) for rel in system.model.candidate_primary_relations],\n''',
    '''        "relations": [_relation_worker_item(rel) for rel in system.model.candidate_primary_relations],\n''',
)
replace_once(
    "src/fusdb/modes/popcon.py",
    '''def _rebuild_system(spec: Mapping[str, Any]) -> Any:\n    from fusdb.profile_sources import source_profile_relation_from_spec\n    from fusdb.registry import RELATIONS, VARIABLES\n''',
    '''def _rebuild_system(spec: Mapping[str, Any]) -> Any:\n    from fusdb.registry import RELATIONS, VARIABLES\n''',
)
replace_once(
    "src/fusdb/modes/popcon.py",
    '''    relations = []\n    for relation_spec in spec["relations"]:\n        if not isinstance(relation_spec, Mapping):\n            raise TypeError(f"Invalid relation rebuild spec {relation_spec!r}.")\n        kind = relation_spec.get("kind")\n        if kind == "registry":\n            relations.append(RELATIONS.get(str(relation_spec["name"])))\n        elif kind == "source_profile":\n            relations.append(source_profile_relation_from_spec(relation_spec))\n        else:\n            raise ValueError(f"Unsupported relation rebuild kind {kind!r}.")\n''',
    '''    relations = [\n        RELATIONS.get(item) if isinstance(item, str) else item\n        for item in spec["relations"]\n    ]\n''',
)

replace_once(
    "tests/test_popcon_source_profiles.py",
    '''    pickle.dumps(spec)\n    generated = [item for item in spec["relations"] if item.get("kind") == "source_profile"]\n    assert len(generated) == 1\n    assert generated[0]["variable"] == "n_e"\n    assert np.asarray(generated[0]["source_values"]).shape == (101,)\n    assert any(\n        item.get("kind") == "registry" and item.get("name") == "Electron density rho-average"\n        for item in spec["relations"]\n    )\n''',
    '''    pickle.dumps(spec)\n    generated = [item for item in spec["relations"] if not isinstance(item, str)]\n    assert len(generated) == 1\n    assert generated[0].source_kind == "source_profile"\n    assert generated[0].source_name == "n_e"\n    assert any(item == "Electron density rho-average" for item in spec["relations"])\n''',
)

# Remove the now-dead Relation worker-rebuild metadata and use dataclasses.replace
# for canonicalized copies, keeping Relation metadata defined in one place.
replace_once(
    "src/fusdb/relation.py",
    "from dataclasses import dataclass, field\n",
    "from dataclasses import dataclass, field, replace\n",
)
replace_once(
    "src/fusdb/relation.py",
    '''        function_name: Decorated Python function name.\n        rebuild_spec: Optional picklable recipe for reconstructing runtime-generated\n            relations in worker processes. Registry relations leave this unset.\n''',
    '''        function_name: Decorated Python function name.\n''',
)
replace_once(
    "src/fusdb/relation.py",
    '''    rebuild_spec: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)\n''',
    "",
)
replace_once(
    "src/fusdb/relation.py",
    '''        self.rebuild_spec = None if self.rebuild_spec is None else dict(self.rebuild_spec)\n''',
    "",
)
replace_once(
    "src/fusdb/relation.py",
    '''        # Local constraints are themselves relations. enforce=False means checked-only applicability.\n        built: list[Relation] = []\n        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):\n            built.append(\n                constraint_from_expression(\n                    text,\n                    name=f"{self.name}_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="relation",\n                    source_name=self.name,\n                )\n            )\n        self.constraint_relations = tuple(built)\n''',
    '''        # Local constraints are themselves relations. enforce=False means checked-only applicability.\n        self.constraint_relations = build_constraint_relations(\n            self.constraints,\n            name_prefix=f"{self.name}_constraint",\n            source_kind="relation",\n            source_name=self.name,\n        )\n''',
)
replace_once(
    "src/fusdb/relation.py",
    '''    return Relation(\n        name=rel.name,\n        func=rel.func,\n        input_names=inputs,\n        outputs=outputs,\n        op=rel.op,\n        rhs=rel.rhs,\n        tags=rel.tags,\n        enforce=rel.enforce,\n        constraints=rel.constraints,\n        source_kind=rel.source_kind,\n        source_name=rel.source_name,\n        constant_names=rel.constant_names,\n        dependency=rel.dependency,\n        function_name=rel.function_name,\n        argument_names=rel.argument_names,\n        rebuild_spec=rel.rebuild_spec,\n    )\n''',
    '''    return replace(rel, input_names=inputs, outputs=outputs)\n''',
)
append_once(
    "src/fusdb/relation.py",
    "def build_constraint_relations(",
    '''def build_constraint_relations(\n    constraints: Any,\n    *,\n    name_prefix: str,\n    source_kind: str,\n    source_name: str,\n) -> tuple[Relation, ...]:\n    """Normalize a constraint declaration into ordinary Relation objects."""\n    return tuple(\n        constraint_from_expression(\n            text,\n            name=f"{name_prefix}_{index}",\n            enforce=enforce,\n            source_kind=source_kind,\n            source_name=source_name,\n        )\n        for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))\n    )\n''',
)

# Phase 5: Variable and VariableSpec use the same constraint normalization helper.
replace_once(
    "src/fusdb/variable.py",
    "from .relation import Relation, constraint_from_expression\n",
    "from .relation import Relation, build_constraint_relations\n",
)
replace_once(
    "src/fusdb/variable.py",
    "from .utils import coerce_numeric_value, coerce_to_shape, parse_constraint_specs, unique_preserve_order, value_in_domain\n",
    "from .utils import coerce_numeric_value, coerce_to_shape, unique_preserve_order, value_in_domain\n",
)
replace_once(
    "src/fusdb/variable.py",
    '''        built: list[Relation] = []\n        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):\n            built.append(\n                constraint_from_expression(\n                    text,\n                    name=f"{self.name}_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="variable",\n                    source_name=self.name,\n                )\n            )\n        object.__setattr__(self, "relations", tuple(built))\n''',
    '''        object.__setattr__(\n            self,\n            "relations",\n            build_constraint_relations(\n                self.constraints,\n                name_prefix=f"{self.name}_constraint",\n                source_kind="variable",\n                source_name=self.name,\n            ),\n        )\n''',
)

replace_once(
    "src/fusdb/registry/variable_registry.py",
    '''            from ..relation import constraint_from_expression\n\n            cached = tuple(\n                constraint_from_expression(\n                    text,\n                    name=f"{self.name}_registry_constraint_{index}",\n                    enforce=enforce,\n                    source_kind="variable",\n                    source_name=self.name,\n                )\n                for index, (text, enforce) in enumerate(self.constraints)\n            )\n''',
    '''            from ..relation import build_constraint_relations\n\n            cached = build_constraint_relations(\n                self.constraints,\n                name_prefix=f"{self.name}_registry_constraint",\n                source_kind="variable",\n                source_name=self.name,\n            )\n''',
)

# The helper script and workflow are transient branch tooling; the successful
# generated commit removes them so production history contains only the refactor.
for transient in (
    ROOT / "tools" / "apply_architecture_refactor.py",
    ROOT / ".github" / "workflows" / "architecture-refactor-apply.yml",
):
    if transient.exists():
        transient.unlink()
