"""One chunk scheduler for every multi-case fusdb operation."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_chunks(
    items: Iterable[T],
    worker: Callable[[tuple[T, ...]], Sequence[R]],
    *,
    workers: int | None = None,
    chunk_size: int | None = None,
) -> list[R]:
    """Apply ``worker`` to ordered chunks and flatten the ordered results.

    This is deliberately only orchestration.  A worker may vectorize its whole
    chunk (POPCON) or run independent scalar solves inside it (the other modes).
    ``chunk_size=None`` means one batch; callers that want automatic parallel
    partitioning should pass :func:`parallel_chunk_size`.
    """
    values = tuple(items)
    if not values:
        return []
    size = len(values) if chunk_size is None else int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive.")
    chunks = [values[start : start + size] for start in range(0, len(values), size)]
    count = 1 if workers in (0, 1) else min(int(workers or (os.cpu_count() or 1)), len(chunks))
    if count == 1:
        return [result for chunk in chunks for result in worker(chunk)]
    with ProcessPoolExecutor(max_workers=count) as executor:
        return [result for results in executor.map(worker, chunks) for result in results]


def parallel_chunk_size(n_items: int, workers: int | None) -> int:
    """Size chunks so each requested worker receives at most one initial chunk."""
    if n_items <= 0:
        return 1
    count = max(1, min(int(workers or (os.cpu_count() or 1)), n_items))
    return max(1, (n_items + count - 1) // count)
