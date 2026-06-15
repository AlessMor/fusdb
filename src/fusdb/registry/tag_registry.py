"""Allowed-tag registry with group-wise matching.

Tag semantics:
    * tags in the same allowed group are OR alternatives;
    * tags in different allowed groups are AND requirements;
    * unknown relation tags are descriptive and ignored for filtering;
    * child reactor tags imply their parent tags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..utils import normalize_tag, normalize_tags


class TagRegistry:
    """Registry of allowed tags and group-wise relation matching."""

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        self.raw = raw or {}
        self.allowed: set[str] = set()
        self.parents: dict[str, set[str]] = {}
        self.tag_to_group: dict[str, str] = {}
        for group, entries in self.raw.items():
            self._load_group(entries, group=normalize_tag(group), parents=())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TagRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle) or {})

    def _register(self, tag: str, *, group: str, parents: tuple[str, ...]) -> None:
        self.allowed.add(tag)
        self.tag_to_group[tag] = group
        self.parents.setdefault(tag, set()).update(parents)

    def _load_group(self, entries: Any, *, group: str, parents: tuple[str, ...]) -> None:
        if entries is None:
            return
        if isinstance(entries, str):
            self._register(normalize_tag(entries), group=group, parents=parents)
            return
        if isinstance(entries, list):
            for item in entries:
                self._load_group(item, group=group, parents=parents)
            return
        if isinstance(entries, dict):
            for parent, children in entries.items():
                tag = normalize_tag(parent)
                self._register(tag, group=group, parents=parents)
                self._load_group(children, group=group, parents=(*parents, tag))
            return
        raise TypeError(f"Invalid allowed-tag entry {entries!r}.")

    def expand(self, tags: Any) -> tuple[str, ...]:
        expanded: set[str] = set()
        for tag in normalize_tags(tags):
            expanded.add(tag)
            expanded.update(self.parents.get(tag, ()))
        return tuple(sorted(expanded))

    def relation_matches(self, relation_tags: Any, reactor_tags: Any) -> bool:
        rel_tags = set(normalize_tags(relation_tags))
        active = set(self.expand(reactor_tags))
        if not rel_tags or not active:
            return True

        required_by_group: dict[str, set[str]] = {}
        for tag in rel_tags:
            group = self.tag_to_group.get(tag)
            if group is None:
                continue
            required_by_group.setdefault(group, set()).add(tag)

        return all(bool(group_tags & active) for group_tags in required_by_group.values())


_DEFAULT_PATH = Path(__file__).with_name("allowed_tags.yaml")
TAGS = TagRegistry.from_yaml(_DEFAULT_PATH) if _DEFAULT_PATH.exists() else TagRegistry({})
