from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable

from fusdb.relation_class import Relation
from fusdb.relation_util import symbol

WarnFunc = Callable[[str, type[Warning] | None], None]

# TAG-BASED DEFAULTS
def default_parameters_for_tags(tags: Iterable[str]) -> dict[str, object]:
    """Return merged defaults for layers that match the provided tags."""
    tag_set = set(tags)
    defaults: dict[str, object] = {}
    defaults["squareness"] = 0.0
    if "tokamak" in tag_set:
        pass
    if "tokamak" in tag_set and "H-mode" in tag_set:
        pass
    # NOTE: other combinations of tags can be added here as needed.
    return defaults


def apply_reactor_defaults(
    parameters: dict[str, Any],
    explicit_parameters: set[str],
    *,
    tags: Iterable[str],
    reactor_id: str,
    warn: WarnFunc | None = None,
) -> tuple[dict[str, object], tuple[Relation, ...]]:
    """Compute parameter defaults and apply loader-level default rules."""
    # Resolve a warning sink so callers can inject structured logging or tests can suppress output.
    warn_func = warn or warnings.warn

    # Start with tag-based defaults, but only keep entries that were not explicitly provided.
    defaults = {
        name: value
        for name, value in default_parameters_for_tags(tags).items()
        if name not in parameters
    }

    fallback_relations: list[Relation] = []

    # If no ion fractions are specified, default to a 50/50 D-T mix with no impurities.
    fraction_keys = ("f_D", "f_T", "f_He3", "f_He4")
    fractions_provided = any(parameters.get(key) is not None for key in fraction_keys)
    if not fractions_provided:
        defaults.setdefault("f_D", 0.5)
        defaults.setdefault("f_T", 0.5)
        defaults.setdefault("f_He3", 0.0)
        defaults.setdefault("f_He4", 0.0)
        # Emit a warning when densities are present, since fractions affect power balance results.
        if parameters.get("n_avg") is not None or parameters.get("n_i") is not None:
            warn_func(
                f"{reactor_id}: ion fractions not provided; assuming 50-50 D-T with no impurities.",
                UserWarning,
            )

    # Build fraction inputs for the steady-state solver, preferring explicit parameters.
    fractions: dict[str, float] = {}
    for key in fraction_keys:
        value = parameters.get(key)
        if value is not None:
            fractions[key] = float(value)
    if not fractions:
        fractions["f_D"] = float(defaults.get("f_D", 0.5))
        fractions["f_T"] = float(defaults.get("f_T", 0.5))

    # Collect particle confinement times, supporting a global tau_p or per-species values.
    tau_p_keys = ("tau_p", "tau_p_D", "tau_p_T", "tau_p_He3", "tau_p_He4")
    tau_p_inputs: dict[str, float] = {}
    for key in tau_p_keys:
        value = parameters.get(key)
        if value is not None:
            tau_p_inputs[key] = float(value)
    has_global_tau = "tau_p" in tau_p_inputs
    if not has_global_tau:
        tau_p_by_fraction = {
            "f_D": "tau_p_D",
            "f_T": "tau_p_T",
            "f_He3": "tau_p_He3",
            "f_He4": "tau_p_He4",
        }
        for frac_key, tau_key in tau_p_by_fraction.items():
            if tau_key not in tau_p_inputs:
                fractions.setdefault(frac_key, 0.0)

    # Infer n_i from n_e (or n_avg) using the charge-weighted fraction relation when needed.
    n_i_value = parameters.get("n_i")
    if n_i_value is None:
        n_e_value = parameters.get("n_e")
        n_e_source = "n_e"
        if n_e_value is None:
            n_e_value = parameters.get("n_avg")
            n_e_source = "n_avg"
        if n_e_value is not None:
            charge_weight = (
                fractions.get("f_D", 0.0)
                + fractions.get("f_T", 0.0)
                + 2.0 * fractions.get("f_He3", 0.0)
                + 2.0 * fractions.get("f_He4", 0.0)
            )
            if charge_weight > 0:
                n_i_value = float(n_e_value) / charge_weight
                parameters["n_i"] = n_i_value
                explicit_parameters.add("n_i")
                defaults.pop("n_i", None)
                if any(key not in fractions for key in fraction_keys):
                    warn_func(
                        f"{reactor_id}: inferred n_i from {n_e_source} with missing fractions treated as zero.",
                        UserWarning,
                    )
                warn_func(
                    f"{reactor_id}: n_i not provided; inferred from {n_e_source}.",
                    UserWarning,
                )

    if n_i_value is not None:
        T_avg_value = parameters.get("T_avg")
        if T_avg_value is None:
            T_avg_value = 0.0
            warn_func(
                f"{reactor_id}: T_avg not provided; assuming 0 keV for composition.",
                UserWarning,
            )
        from fusdb.relations.plasma_composition import solve_steady_state_composition

        # Run the steady-state composition solver whenever we have total ion density.
        steady_fractions = solve_steady_state_composition(
            n_i=float(n_i_value),
            T_avg=float(T_avg_value),
            fractions=fractions,
            tau_p=tau_p_inputs or None,
        )
        for key, value in steady_fractions.items():
            parameters[key] = value
            explicit_parameters.add(key)
            defaults.pop(key, None)
    else:
        warn_func(
            f"{reactor_id}: n_i not provided; skipping steady-state composition.",
            UserWarning,
        )

    # If n_la is missing, add a low-priority fallback relation tied to n_avg.
    if parameters.get("n_la") is None:
        fallback_relations.append(
            Relation(
                "Line-averaged density fallback",
                ("n_la", "n_avg"),
                symbol("n_la") - symbol("n_avg"),
                priority=-100,
                solve_for=("n_la",),
            )
        )
        warn_func(
            f"{reactor_id}: n_la not provided; will fall back to n_avg when available.",
            UserWarning,
        )
    return defaults, tuple(fallback_relations)
