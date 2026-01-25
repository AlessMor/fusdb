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

    # If n_avg is missing but P_fus is available, compute n_avg backwards from fusion power.
    # This calculation runs in reactor_defaults AFTER relations have set defaults/computed values.
    # The bidirectional solver + existing relations should handle this, but the quadratic inverse
    # through Rr_DT requires that V_p be available, which may not be the case initially.
    if parameters.get("n_avg") is None and parameters.get("P_fus") is not None:
        P_fus_val = parameters.get("P_fus")
        T_avg_val = parameters.get("T_avg")
        V_p_val = parameters.get("V_p")
        
        if T_avg_val is not None and V_p_val is not None:
            from fusdb.relations.power_balance.fusion_power.reactivity_functions import sigmav_DT_BoschHale
            from fusdb.registry.constants import KEV_TO_J
            
            # Get fractions (use defaults if not set)
            f_D_val = parameters.get("f_D", defaults.get("f_D", 0.5))
            f_T_val = parameters.get("f_T", defaults.get("f_T", 0.5))
            
            # DT fusion energy: 3.5 MeV (alpha) + 14.1 MeV (neutron) = 17.6 MeV
            E_total = 17.6e3 * KEV_TO_J  # Convert to Joules
            
            # Get DT reactivity at T_avg
            sigmav = sigmav_DT_BoschHale(float(T_avg_val))
            
            # Assume P_fus ≈ P_fus_DT (DT dominates for D-T plasmas)
            # Solve for n_i: n_i = sqrt(P_fus / (f_D × f_T × σv × V_p × E_total))
            import math
            denominator = float(f_D_val) * float(f_T_val) * sigmav * float(V_p_val) * E_total
            if denominator > 0:
                n_i_from_P_fus = math.sqrt(float(P_fus_val) / denominator)
                
                # n_avg ≈ n_i for quasineutrality
                parameters["n_avg"] = n_i_from_P_fus
                explicit_parameters.add("n_avg")
                defaults.pop("n_avg", None)
                
                # Immediately compute n_i from n_avg using quasineutrality
                # n_e = n_avg and n_e = n_i * (f_D + f_T + 2*f_He3 + 2*f_He4)
                f_He3_val = parameters.get("f_He3", defaults.get("f_He3", 0.0))
                f_He4_val = parameters.get("f_He4", defaults.get("f_He4", 0.0))
                charge_weight = float(f_D_val) + float(f_T_val) + 2.0 * float(f_He3_val) + 2.0 * float(f_He4_val)
                
                if charge_weight > 0 and parameters.get("n_i") is None:
                    parameters["n_i"] = float(parameters["n_avg"]) / charge_weight
                    explicit_parameters.add("n_i")
                    defaults.pop("n_i", None)
                    
                    # Also set n_e from quasineutrality
                    if parameters.get("n_e") is None:
                        parameters["n_e"] = parameters["n_avg"]
                        explicit_parameters.add("n_e")
                        defaults.pop("n_e", None)

    # Infer n_i from n_e (or n_avg) using the charge-weighted fraction relation when needed.
    # Keep inferred values local so the relation system can still reconcile n_i later.
    n_i_value = parameters.get("n_i")
    n_i_seed = n_i_value
    if n_i_seed is None:
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
                n_i_seed = float(n_e_value) / charge_weight
                if any(key not in fractions for key in fraction_keys):
                    warn_func(
                        (
                            f"{reactor_id}: inferred n_i from {n_e_source} for composition "
                            "with missing fractions treated as zero."
                        ),
                        UserWarning,
                    )
                warn_func(
                    f"{reactor_id}: n_i not provided; inferred from {n_e_source} for composition.",
                    UserWarning,
                )

    if n_i_seed is not None:
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
            n_i=float(n_i_seed),
            T_avg=float(T_avg_value),
            fractions=fractions,
            tau_p=tau_p_inputs or None,
        )
        for key, value in steady_fractions.items():
            parameters[key] = value
            explicit_parameters.add(key)
            defaults.pop(key, None)
        
        # Also set n_i itself, since the composition solver only returns fractions
        if "n_i" not in parameters:
            parameters["n_i"] = n_i_seed
            explicit_parameters.add("n_i")
            defaults.pop("n_i", None)
        
        # Compute species densities from fractions and n_i
        for species_key, fraction_key in [("n_D", "f_D"), ("n_T", "f_T"), ("n_He3", "f_He3"), ("n_He4", "f_He4")]:
            if species_key not in parameters:
                fraction = parameters.get(fraction_key, defaults.get(fraction_key, 0.0))
                if fraction is not None:
                    parameters[species_key] = float(fraction) * float(n_i_seed)
                    explicit_parameters.add(species_key)
                    defaults.pop(species_key, None)
        
        # Compute reaction rates if we have all required parameters
        # This is a workaround for solver not picking up newly enabled relations
        T_avg_for_rr = parameters.get("T_avg")
        V_p_for_rr = parameters.get("V_p")
        if T_avg_for_rr is not None and V_p_for_rr is not None and n_i_seed is not None:
            from fusdb.relations.power_balance.fusion_power.reactivity_functions import sigmav_DT_BoschHale
            
            # Compute Rr_DT if not already set
            if "Rr_DT" not in parameters:
                f_D_rr = parameters.get("f_D", defaults.get("f_D", 0.5))
                f_T_rr = parameters.get("f_T", defaults.get("f_T", 0.5))
                if f_D_rr is not None and f_T_rr is not None:
                    reactivity = sigmav_DT_BoschHale(float(T_avg_for_rr))
                    n_D_val = float(f_D_rr) * float(n_i_seed)
                    n_T_val = float(f_T_rr) * float(n_i_seed)
                    Rr_DT_val = n_D_val * n_T_val * reactivity * float(V_p_for_rr)
                    parameters["Rr_DT"] = Rr_DT_val
                    explicit_parameters.add("Rr_DT")
                    defaults.pop("Rr_DT", None)
                    
                    # Also compute P_fus_DT components from Rr_DT
                    # E_DT_alpha = 3.5 MeV, E_DT_n = 14.1 MeV
                    from fusdb.registry.constants import KEV_TO_J
                    E_alpha = 3.5e3 * KEV_TO_J  # 3.5 MeV in Joules
                    E_n = 14.1e3 * KEV_TO_J      # 14.1 MeV in Joules
                    
                    if "P_fus_DT_alpha" not in parameters:
                        parameters["P_fus_DT_alpha"] = Rr_DT_val * E_alpha
                        explicit_parameters.add("P_fus_DT_alpha")
                        defaults.pop("P_fus_DT_alpha", None)
                    
                    if "P_fus_DT_n" not in parameters:
                        parameters["P_fus_DT_n"] = Rr_DT_val * E_n
                        explicit_parameters.add("P_fus_DT_n")
                        defaults.pop("P_fus_DT_n", None)
                    
                    if "P_fus_DT" not in parameters:
                        parameters["P_fus_DT"] = Rr_DT_val * (E_alpha + E_n)
                        explicit_parameters.add("P_fus_DT")
                        defaults.pop("P_fus_DT", None)
    else:
        warn_func(
            f"{reactor_id}: n_i not provided; skipping steady-state composition.",
            UserWarning,
        )

    # If n_i is missing but n_avg is available, compute it directly from quasineutrality
    # This ensures n_i, n_e, T_e, T_i are available for thermal pressure calculations
    if parameters.get("n_i") is None and parameters.get("n_avg") is not None:
        # Get fractions (use defaults if not in parameters)
        f_D_val = parameters.get("f_D", defaults.get("f_D", 0.5))
        f_T_val = parameters.get("f_T", defaults.get("f_T", 0.5))
        f_He3_val = parameters.get("f_He3", defaults.get("f_He3", 0.0))
        f_He4_val = parameters.get("f_He4", defaults.get("f_He4", 0.0))
        
        # Quasineutrality: n_e = n_avg and n_e = n_i * (f_D + f_T + 2*f_He3 + 2*f_He4)
        # Therefore: n_i = n_avg / charge_weight
        charge_weight = f_D_val + f_T_val + 2.0 * f_He3_val + 2.0 * f_He4_val
        
        if charge_weight > 0:
            n_i_computed = float(parameters["n_avg"]) / float(charge_weight)
            parameters["n_i"] = n_i_computed
            explicit_parameters.add("n_i")
            defaults.pop("n_i", None)
            
            # Also set n_e from quasineutrality (n_e ≈ n_avg for quasineutral plasma)
            if parameters.get("n_e") is None:
                parameters["n_e"] = parameters["n_avg"]
                explicit_parameters.add("n_e")
                defaults.pop("n_e", None)
    
    # If T_e or T_i are missing but T_avg is available, assume T_e = T_i = T_avg
    # This is a common approximation for thermal equilibrium
    T_avg_val = parameters.get("T_avg")
    if T_avg_val is not None:
        if parameters.get("T_e") is None:
            parameters["T_e"] = T_avg_val
            explicit_parameters.add("T_e")
            defaults.pop("T_e", None)
        if parameters.get("T_i") is None:
            parameters["T_i"] = T_avg_val
            explicit_parameters.add("T_i")
            defaults.pop("T_i", None)

    # If n_la is missing, add a fallback relation tied to n_avg.
    if parameters.get("n_la") is None:
        fallback_relations.append(
            Relation(
                "Line-averaged density fallback",
                ("n_la", "n_avg"),
                symbol("n_la") - symbol("n_avg"),
                solve_for=("n_la",),
            )
        )
        warn_func(
            f"{reactor_id}: n_la not provided; will fall back to n_avg when available.",
            UserWarning,
        )
    return defaults, tuple(fallback_relations)
