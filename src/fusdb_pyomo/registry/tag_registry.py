"""Tag registry and matching rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..utils import normalize_tag, normalize_tags


class TagRegistry:
    """Registry of allowed tags with simple parent-child expansion."""

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        self.raw = raw or {}
        self.parents: dict[str, set[str]] = {}
        self.allowed: set[str] = set()
        for group, entries in self.raw.items():
            self._load_group(entries, parents=())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TagRegistry":
        """Load tags from YAML."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle) or {})

    def _load_group(self, entries: Any, *, parents: tuple[str, ...]) -> None:
        if entries is None:
            return
        if isinstance(entries, str):
            tag = normalize_tag(entries)
            self.allowed.add(tag)
            self.parents.setdefault(tag, set()).update(parents)
            return
        if isinstance(entries, list):
            for item in entries:
                self._load_group(item, parents=parents)
            return
        if isinstance(entries, dict):
            for parent, children in entries.items():
                p = normalize_tag(parent)
                self.allowed.add(p)
                self.parents.setdefault(p, set()).update(parents)
                self._load_group(children, parents=(*parents, p))

    def expand(self, tags: Any) -> tuple[str, ...]:
        """Return tags plus all parent tags."""
        expanded: set[str] = set()
        for tag in normalize_tags(tags):
            expanded.add(tag)
            expanded.update(self.parents.get(tag, ()))
        return tuple(sorted(expanded))

    def relation_matches(self, relation_tags: Any, reactor_tags: Any) -> bool:
        """Return whether a relation should be active for reactor tags."""
        rel_tags = set(normalize_tags(relation_tags))
        if not rel_tags:
            return True
        active = set(self.expand(reactor_tags))
        if not active:
            return True
        # Tags unknown to the registry are treated as descriptive and do not filter.
        required = {tag for tag in rel_tags if tag in self.allowed}
        return required <= active


_DEFAULT_PATH = Path(__file__).with_name("allowed_tags.yaml")
TAGS = TagRegistry.from_yaml(_DEFAULT_PATH) if _DEFAULT_PATH.exists() else TagRegistry({})
