"""
Reactor defaults 

NOTE: If only some ion fractions are defined (e.g., f_D but not f_T),
the solver may produce incorrect results. Always define all fractions
or none (to get 50-50 D-T default).

NOTE: The 0D assumption (n_la = n_avg) is a simplification for 0D values
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

from fusdb.relation_class import Relation
from fusdb.relation_util import symbol

WarnFunc = Callable[[str, type[Warning] | None], None]


def apply_reactor_defaults(
    parameters: dict[str, Any],
    explicit_parameters: set[str],
    *,
    tags: set[str],
    reactor_id: str,
    warn: WarnFunc | None = None,
) -> tuple[dict[str, object], tuple[Relation, ...]]:
    """Apply default values and return fallback relations.
    
    Returns:
        (defaults_dict, fallback_relations): Fixed defaults and bidirectional relations.
    """
    warn_func = warn or warnings.warn
    defaults: dict[str, object] = {}
    fallback_relations: list[Relation] = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. GEOMETRY DEFAULTS
    # ═══════════════════════════════════════════════════════════════════════════
    if "squareness" not in parameters:
        defaults["squareness"] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. D-T OPERATION (50-50 mix)
    # ═══════════════════════════════════════════════════════════════════════════
    fraction_keys = ("f_D", "f_T", "f_He3", "f_He4")
    defined_fractions = [k for k in fraction_keys if parameters.get(k) is not None]
    
    if not defined_fractions:
        # No fractions defined: use 50-50 D-T default
        defaults["f_D"] = 0.5
        defaults["f_T"] = 0.5
        defaults["f_He3"] = 0.0
        defaults["f_He4"] = 0.0
        if parameters.get("n_avg") is not None or parameters.get("n_i") is not None:
            warn_func(f"{reactor_id}: ion fractions not provided; assuming 50-50 D-T.", UserWarning)
    elif len(defined_fractions) < len(fraction_keys):
        # Partial fractions defined: warn about potential inconsistency
        missing = [k for k in fraction_keys if k not in defined_fractions]
        warn_func(
            f"{reactor_id}: partial ion fractions defined ({defined_fractions}), "
            f"missing {missing}. Solver may give incorrect results.",
            UserWarning
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. NO PROFILES (0D assumption): n_la = n_avg
    # ═══════════════════════════════════════════════════════════════════════════
    # Always add this as fallback - needed for tau_E scalings when n_la isn't provided
    if parameters.get("n_la") is None:
        fallback_relations.append(
            Relation("n_la = n_avg (0D)", ("n_la", "n_avg"),
                     symbol("n_la") - symbol("n_avg"), solve_for=("n_la", "n_avg"))
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. QUASINEUTRALITY: n_e = n_avg (simplified, Z_eff ~ 1)
    # ═══════════════════════════════════════════════════════════════════════════
    if parameters.get("n_e") is None or parameters.get("n_avg") is None:
        fallback_relations.append(
            Relation("n_e = n_avg (quasineutrality)", ("n_e", "n_avg"),
                     symbol("n_e") - symbol("n_avg"), solve_for=("n_e", "n_avg"))
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. ISOTHERMAL APPROXIMATION: T_i = T_e = T_avg
    # ═══════════════════════════════════════════════════════════════════════════
    if parameters.get("T_e") is None or parameters.get("T_avg") is None:
        fallback_relations.append(
            Relation("T_e = T_avg (isothermal)", ("T_e", "T_avg"),
                     symbol("T_e") - symbol("T_avg"), solve_for=("T_e", "T_avg")))
    if parameters.get("T_i") is None or parameters.get("T_avg") is None:
        fallback_relations.append(
            Relation("T_i = T_avg (isothermal)", ("T_i", "T_avg"),
                     symbol("T_i") - symbol("T_avg"), solve_for=("T_i", "T_avg")))
    
    return defaults, tuple(fallback_relations)
