from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from fusiondb.relations_util import REL_TOL_DEFAULT, coerce_number

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


@dataclass
class Relation:
    """Implicit relation defined by a zero-residual equation."""

    name: str
    variables: tuple[str, ...]
    equation: Callable[[Mapping[str, float]], float]
    priority: int | None = None
    rel_tol: float = REL_TOL_DEFAULT
    solve_for: tuple[str, ...] | None = None
    initial_guesses: Mapping[str, float | Callable[[Mapping[str, float]], float]] | None = None
    max_solve_iterations: int = 25

    def with_tol(self, rel_tol: float, priority: int | None = None) -> Relation:
        return Relation(
            name=self.name,
            variables=self.variables,
            equation=self.equation,
            priority=self.priority if priority is None else priority,
            rel_tol=rel_tol,
            solve_for=self.solve_for,
            initial_guesses=self.initial_guesses,
            max_solve_iterations=self.max_solve_iterations,
        )

    def apply(self, var_map: dict[str, Variable], warn: WarnFunc) -> bool:
        changed = False
        targets = self.solve_for or self.variables
        for target in targets:
            inputs = [v for v in self.variables if v != target]
            if any(var_map.get(i) is None or var_map[i].value is None for i in inputs):
                continue
            known = {i: var_map[i].value for i in inputs}  # type: ignore[arg-type]
            guess = (self.initial_guesses or {}).get(target)
            # inline seed + secant solve to avoid extra helpers
            seed = guess(known) if callable(guess) else guess
            if seed is None or not math.isfinite(seed):
                seed = known.get(target)
            if seed is None or not math.isfinite(seed):
                mags = [abs(v) for v in known.values() if v is not None and math.isfinite(v)]
                seed = max(mags[0], 1e-3) if mags else 1.0
            x0 = float(seed)
            x1 = x0 * 1.1 if x0 != 0 else 1.0

            def residual(x: float) -> float:
                trial = dict(known)
                trial[target] = x
                return float(self.equation(trial))

            f0, f1 = residual(x0), residual(x1)
            eps = 1e-12
            scale = max([abs(v) for v in known.values() if v is not None and math.isfinite(v)] + [1.0])
            max_mag = scale * 1e6
            result = x1
            for _ in range(self.max_solve_iterations):
                if math.isfinite(f1) and abs(f1) < eps:
                    result = x1
                    break
                denom = f1 - f0
                if denom == 0:
                    step = 1.0 if x1 == 0 else max(abs(x1) * 0.1, 1e-3)
                    x1 += step
                    f0, f1 = f1, residual(x1)
                    continue
                nxt = x1 - f1 * (x1 - x0) / denom
                if not math.isfinite(nxt):
                    nxt = x1 + 1.0
                if abs(nxt) > max_mag:
                    nxt = math.copysign(max_mag, nxt)
                if math.isclose(nxt, x1, rel_tol=1e-12, abs_tol=1e-12):
                    result = nxt
                    break
                x0, f0 = x1, f1
                x1 = nxt
                f1 = residual(x1)
                result = x1

            var = var_map.setdefault(target, Variable(target))
            changed |= var.assign(
                result,
                self.priority if self.priority is not None else PRIORITY_RELATION,
                self.name,
                self.rel_tol,
                warn,
            )
        return changed


class RelationSystem:
    """Minimal wrapper to solve a set of relations as a system."""

    def __init__(self, relations: Sequence[Relation], *, rel_tol: float = REL_TOL_DEFAULT, warn: WarnFunc = warnings.warn) -> None:
        self.relations = relations
        self.rel_tol = rel_tol
        self.warn_sink = warn
        self.warn: WarnFunc = self._collect_warn
        self.var_map: dict[str, Variable] = {}
        self._pending_warnings: list[tuple[str, type[Warning] | None]] = []
        self._seen_warnings: set[tuple[str, type[Warning] | None]] = set()

    @property
    def values(self) -> dict[str, float | None]:
        return {name: var.value for name, var in self.var_map.items()}

    def set(self, name: str, value: float | None, *, priority: int = PRIORITY_EXPLICIT, source: str = "explicit") -> None:
        var = self.var_map.setdefault(name, Variable(name))
        number = coerce_number(value, name)
        if number is None:
            return
        var.assign(number, priority, source, self.rel_tol, self.warn)

    def _collect_warn(self, message: str, category: type[Warning] | None = None) -> None:
        key = (message, category)
        if key in self._seen_warnings:
            return
        self._seen_warnings.add(key)
        self._pending_warnings.append(key)

    def _flush_warnings(self) -> None:
        emitted_vars: set[str] = set()
        for message, category in self._pending_warnings:
            var_name = message.split(" ", 1)[0]
            if var_name in emitted_vars:
                continue
            emitted_vars.add(var_name)
            self.warn_sink(message, UserWarning if category is None else category)
        self._pending_warnings.clear()
        self._seen_warnings.clear()

    def solve(self, *, max_iterations: int = 50) -> dict[str, float | None]:
        self._pending_warnings.clear()
        self._seen_warnings.clear()
        for _ in range(max_iterations):
            changed = False
            for rel in self.relations:
                changed |= rel.apply(self.var_map, self._collect_warn)
            if not changed:
                break
        self._flush_warnings()
        return self.values
