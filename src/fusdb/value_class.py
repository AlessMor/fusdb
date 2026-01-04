from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

WarnFunc = Callable[[str, type[Warning] | None], None]

PRIORITY_EXPLICIT = 100
PRIORITY_RELATION = 10
PRIORITY_STRICT = 120


@dataclass
class Variable:
    name: str
    value: float | None = None
    priority: int = PRIORITY_RELATION
    source: str = "unknown"
    _warned_sources: set[str] = field(default_factory=set, init=False, repr=False)

    def assign(self, candidate: float, priority: int, source: str, rel_tol: float, warn: WarnFunc) -> bool:
        if candidate is None or not math.isfinite(candidate):
            return False

        def warn_once(message: str) -> None:
            if source in self._warned_sources:
                return
            self._warned_sources.add(source)
            warn(message, UserWarning)

        if self.value is None:
            self.value, self.priority, self.source = candidate, priority, source
            return True

        if math.isclose(self.value, candidate, rel_tol=rel_tol, abs_tol=0.0):
            if candidate != self.value:
                warn_once(
                    f"{self.name} differs from {source} but is within tolerance: existing {self.value}, new {candidate}"
                )
            if priority > self.priority:
                self.priority, self.source = priority, source
                return True
            return False

        if priority > self.priority:
            warn_once(f"{self.name} updated from {self.value} ({self.source}) to {candidate} ({source})")
            self.value, self.priority, self.source = candidate, priority, source
            return True

        warn_once(f"Inconsistent {self.name}: keeping {self.value} ({self.source}) over {candidate} ({source})")
        return False


__all__ = [
    "PRIORITY_EXPLICIT",
    "PRIORITY_RELATION",
    "PRIORITY_STRICT",
    "Variable",
]
