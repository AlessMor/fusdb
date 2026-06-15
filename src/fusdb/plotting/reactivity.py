"""Fusion reactivity curves, ``<sigma v>`` versus ion temperature.

Reproduces the representation in ``examples/reactivity_plots.ipynb`` as a
reusable plotter. Reaction sources are discovered from the relation registry, so
new reactions and parametrisations appear without editing this module.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from matplotlib.axes import Axes

from fusdb.registry import RELATIONS

from .style import axes

# Preferred parametrisation per reaction, best-first; mirrors the registry
# ``default_method`` values (Bosch-Hale where available, otherwise CF88).
METHOD_PREFERENCE = ("BoschHale", "CF88", "Hively", "NRL", "ENDFB-VIII1", "ENDFB-VIII0")
DEFAULT_TEMPERATURE_KEV = np.logspace(0.0, 2.7, 240)

# A reactivity source is either a ``@relation`` object or a plain ``f(T_i=...)``.
ReactivitySource = Callable[..., Any]


def _method(relation: Any) -> str:
    """Return the parametrisation name, e.g. ``"BoschHale"`` (last name token)."""
    return relation.name.split()[-1]


def _reaction(relation: Any) -> str:
    """Return the reaction label, e.g. ``"DT"`` (first name token)."""
    return relation.name.split()[0]


def _evaluate(source: ReactivitySource, temperature_keV: np.ndarray) -> np.ndarray:
    """Evaluate a reactivity source on a temperature grid as a float array."""
    if hasattr(source, "evaluate"):  # Relation object
        values = source.evaluate({"T_i": temperature_keV})
    else:  # plain callable f(T_i=...)
        values = source(T_i=temperature_keV)
    return np.asarray(values, dtype=float)


def default_reactivities() -> dict[str, ReactivitySource]:
    """Return one preferred reactivity relation per reaction, keyed by label.

    Discovered from the registry: every relation mapping ``T_i`` to a single
    ``sigmav_*`` output. For each reaction the method earliest in
    :data:`METHOD_PREFERENCE` is kept.

    Returns:
        Mapping of ``"<reaction> (<method>)"`` to the chosen relation.
    """
    by_output: dict[str, list[Any]] = {}
    for relation in RELATIONS:
        outputs = relation.outputs
        if relation.input_names == ("T_i",) and len(outputs) == 1 and outputs[0].startswith("sigmav_"):
            by_output.setdefault(outputs[0], []).append(relation)

    def rank(relation: Any) -> int:
        method = _method(relation)
        return METHOD_PREFERENCE.index(method) if method in METHOD_PREFERENCE else len(METHOD_PREFERENCE)

    chosen: dict[str, ReactivitySource] = {}
    for _output, candidates in sorted(by_output.items()):
        best = min(candidates, key=rank)
        chosen[f"{_reaction(best)} ({_method(best)})"] = best
    return chosen


def plot_reactivity(
    reactions: Mapping[str, ReactivitySource] | None = None,
    *,
    temperature_keV: np.ndarray | None = None,
    ax: Axes | None = None,
    **plot_kw: Any,
) -> Axes:
    """Plot fusion reactivities on a log-log axis.

    Args:
        reactions: Mapping of legend label -> reactivity source (a ``@relation``
            object or any ``f(T_i=...)`` callable). Defaults to one curve per
            reaction from :func:`default_reactivities`.
        temperature_keV: Ion-temperature grid in keV. Defaults to ~1-500 keV.
        ax: Existing axis to draw on; a new figure is created when omitted.
        **plot_kw: Forwarded to ``Axes.loglog`` (e.g. ``linewidth``).

    Returns:
        The axis the curves were drawn on.
    """
    reactions = dict(reactions) if reactions is not None else default_reactivities()
    if temperature_keV is None:
        temperature_keV = DEFAULT_TEMPERATURE_KEV
    else:
        temperature_keV = np.asarray(temperature_keV, dtype=float)

    ax = axes(ax, figsize=(10, 6))
    plot_kw.setdefault("linewidth", 2)
    for label, source in reactions.items():
        ax.loglog(temperature_keV, _evaluate(source, temperature_keV), label=label, **plot_kw)

    ax.set_xlabel("Ion temperature [keV]")
    ax.set_ylabel(r"$\langle \sigma v \rangle$ [m$^3$/s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax
